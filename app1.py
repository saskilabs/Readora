from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import pandas as pd
from datetime import datetime, date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "readora_cozy_secret_key_123"

app.config.update(
    SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True, )

db_url = os.environ.get("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = { "pool_pre_ping": True, "pool_recycle": 300, }

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app) 

# Model Database
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    ratings = db.relationship('UserRating', backref='user', lazy=True)
    daily_moods = db.relationship("DailyMood",backref="user",lazy=True)

class Book(db.Model):
    __tablename__ = 'books'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    author = db.Column(db.String(255))
    genre = db.Column(db.String(100), index=True)
    rating = db.Column(db.Float)
    tahun_terbit = db.Column(db.Integer)
    halaman = db.Column(db.Integer)
    mood_utama = db.Column(db.String(100))
    mood_pendukung1 = db.Column(db.String(100))
    mood_pendukung2 = db.Column(db.String(100))
    cover = db.Column(db.String(255))
    sinopsis = db.Column(db.Text)

    ratings = db.relationship('UserRating', backref='book', lazy=True)

class UserRating(db.Model):
    __tablename__ = 'user_ratings'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    book_id = db.Column(db.Integer, db.ForeignKey('books.id'), nullable=False, index=True)
    rating = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

class DailyMood(db.Model):
    __tablename__ = "daily_moods"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer,db.ForeignKey("users.id"),nullable=False, index=True)
    mood = db.Column(db.String(100), nullable=False)
    selected_date = db.Column(db.Date,nullable=False)

with app.app_context():
    db.create_all()

    # Isi database kalau masih kosong
    if Book.query.count() == 0:
        df = pd.read_csv("dataset/DATASETBUKUFINAL.csv")

        for _, row in df.iterrows():
            book = Book(
                title=row["Judul"],
                author=row["Penulis"],
                genre=row["Genre"],
                rating=float(str(row["Rating"]).replace(",",".")),
                tahun_terbit=row["Tahun Terbit"],
                halaman=row["Halaman"],
                mood_utama=row["Mood Utama"],
                mood_pendukung1=row["Mood Pendukung1"],
                mood_pendukung2=row["Mood Pendukung2"],
                cover=row["Cover"],
                sinopsis=row["Sinopsis"]
            )

            db.session.add(book)

        db.session.commit()
    print("Jumlah buku:", Book.query.count())

# TF-IDF CACHE
tfidf_cache = {}

def _build_content(b):
    return (
        (str(b.mood_utama) + " ") * 5 +
        (str(b.mood_pendukung1) + " ") * 2 +
        str(b.mood_pendukung2)
    )

def rebuild_tfidf_cache():
    """Hitung ulang TF-IDF untuk semua genre dan simpan ke cache."""
    tfidf_cache.clear()

    genres = [g[0] for g in db.session.query(Book.genre).distinct().all()]

    for genre in genres:
        books = Book.query.filter_by(genre=genre).all()
        if not books:
            continue

        contents = [_build_content(b) for b in books]

        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(contents)

        tfidf_cache[genre] = {
            "vectorizer": tfidf,
            "matrix": matrix,
            "books": books,  # urutan sejajar dengan baris matrix
        }

with app.app_context():
    rebuild_tfidf_cache()
    print("TF-IDF cache siap untuk", len(tfidf_cache), "genre")


def _book_to_dict(b):
    return {
        "ID": b.id,
        "Judul": b.title,
        "Penulis": b.author,
        "Genre": b.genre,
        "Rating": b.rating,
        "Halaman": b.halaman,
        "Tahun Terbit": b.tahun_terbit,
        "Mood Utama": b.mood_utama,
        "Mood Pendukung1": b.mood_pendukung1,
        "Mood Pendukung2": b.mood_pendukung2,
        "Cover": b.cover,
        "Sinopsis": b.sinopsis,
    }


def recommend_books(genre_input, mood_input, top_n=5):
    cache = tfidf_cache.get(genre_input)
    if not cache:
        return None

    books = cache["books"]
    tfidf = cache["vectorizer"]
    matrix = cache["matrix"]

    mood_list = mood_input.split()

    # Validasi apakah mood yang dicari ada di dalam data genre ini
    mood_exists = any(
        b.mood_utama in mood_list or
        b.mood_pendukung1 in mood_list or
        b.mood_pendukung2 in mood_list
        for b in books
    )
    if not mood_exists:
        return None

    query_vector = tfidf.transform([mood_input])
    scores = cosine_similarity(query_vector, matrix).flatten()

    valid_indices = [i for i, score in enumerate(scores) if score > 0]
    if not valid_indices:
        return None

    top_indices = sorted(valid_indices, key=lambda x: scores[x], reverse=True)[:top_n]

    return [_book_to_dict(books[i]) for i in top_indices]


def recommend_similar_books(book_id, top_n=5):
    selected_book = Book.query.get(book_id)
    if not selected_book:
        return None

    genre = selected_book.genre
    cache = tfidf_cache.get(genre)
    if not cache:
        return None

    books = cache["books"]
    tfidf = cache["vectorizer"]
    matrix = cache["matrix"]

    mood_query = " ".join(filter(None, [
        selected_book.mood_utama,
        selected_book.mood_pendukung1,
        selected_book.mood_pendukung2
    ]))

    query_vector = tfidf.transform([mood_query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    # exclude buku yang sedang dilihat
    valid_indices = [
        i for i, score in enumerate(scores)
        if score > 0 and books[i].id != book_id
    ]
    if not valid_indices:
        return None

    top_indices = sorted(valid_indices, key=lambda x: scores[x], reverse=True)[:top_n]

    return [_book_to_dict(books[i]) for i in top_indices]


def recommend_by_favorites(user_id, top_n=6, min_rating=4):
    """
    Rekomendasi personal berdasarkan buku-buku yang pernah
    dirating tinggi (>= min_rating) oleh user.
    """
    favorite_ratings = (
        UserRating.query
        .filter_by(user_id=user_id)
        .filter(UserRating.rating >= min_rating)
        .all()
    )
    if not favorite_ratings:
        return None

    rated_book_ids = {
        r.book_id for r in UserRating.query.filter_by(user_id=user_id).all()
    }

    score_map = {}
    book_cache = {}

    for fav in favorite_ratings:
        fav_book = fav.book
        if not fav_book:
            continue

        genre = fav_book.genre
        cache = tfidf_cache.get(genre)
        if not cache:
            continue

        books = cache["books"]
        tfidf = cache["vectorizer"]
        matrix = cache["matrix"]

        mood_query = " ".join(filter(None, [
            fav_book.mood_utama,
            fav_book.mood_pendukung1,
            fav_book.mood_pendukung2
        ]))

        query_vector = tfidf.transform([mood_query])
        scores = cosine_similarity(query_vector, matrix).flatten()

        for i, score in enumerate(scores):
            if score <= 0:
                continue
            b = books[i]
            if b.id in rated_book_ids:
                continue
            weighted_score = score * fav.rating
            score_map[b.id] = score_map.get(b.id, 0) + weighted_score
            book_cache[b.id] = b

    if not score_map:
        return None

    top_ids = sorted(score_map, key=lambda k: score_map[k], reverse=True)[:top_n]
    return [_book_to_dict(book_cache[bid]) for bid in top_ids]


@app.route("/")
def home():
    print("HOME SESSION:", session)
    if not session.get("user"):
        return render_template("landing.html", active_page="home")
    
    today = date.today()
    
    # Ambil data mood hari ini
    daily_mood = DailyMood.query.filter_by(
        user_id =session["user_id"],
        selected_date=today
    ).first()

    #kalau user sudah memilih mood hari ini
    if daily_mood:
        selected_mood = daily_mood.mood

        genres = [
            "Romance",
            "Mystery/Thriller",
            "Fantasy",
            "Sci-Fi",
            "Self Improvement"
        ]

        recommendations_by_genre = {}

        for genre in genres:
            books = recommend_books(genre, selected_mood, top_n=2)
            if books:
                # BARU: tambahin UserRating
                user_id = session["user_id"]
                for b in books:
                    b["UserRating"] = None
                    existing = UserRating.query.filter_by(user_id=user_id, book_id=b["ID"]).first()
                    if existing:
                        b["UserRating"] = existing.rating
                recommendations_by_genre[genre] = books

        return render_template(
            "dashbord.html",
            mood=selected_mood,
            genres_data=recommendations_by_genre,
            active_page="home"
        )
    
    #kalau belum milih mood harian
    moods_list = [
        "Sad",
        "Emotional",
        "Reflective",
        "Heartwarming",
        "Suspenseful",
        "Happy/Funny",
        "Hopeful",
        "Thrilling",
        "Dark",
        "Inspiring"
    ]

    return render_template(
        "select_mood.html",
        moods=moods_list,
        active_page="home"
    )

@app.context_processor
def profile_data():

    if not session.get("user_id"):
        return {}

    today = date.today()

    daily = DailyMood.query.filter_by(
        user_id=session["user_id"],
        selected_date=today
    ).first()

    current_mood = daily.mood if daily else "-"

    ratings = (
        UserRating.query
        .filter_by(user_id=session["user_id"])
        .order_by(UserRating.created_at.desc())
        .limit(5)
        .all()
    )

    total_ratings = UserRating.query.filter_by(
        user_id=session["user_id"]
    ).count()

    return dict(
        current_mood=current_mood,
        rating_history=ratings,
        total_ratings=total_ratings
    )
    
@app.route("/select_mood", methods=["POST"])
def select_mood():
    if not session.get("user"):
        return redirect(url_for("login"))
    
    mood = request.form.get("mood")
    if mood:
        today = date.today()

        existing = DailyMood.query.filter_by(
            user_id=session["user_id"],
            selected_date=today
        ).first()

        if existing:
            existing.mood = mood
        else:
            new_mood = DailyMood(
                user_id=session["user_id"],
                mood=mood,
                selected_date=today
            )
            db.session.add(new_mood)

        db.session.commit()

    return redirect(url_for("home"))

@app.route("/rate_book", methods=["POST"])
def rate_book():
    if not session.get("user_id"):
        return redirect(url_for("login"))

    book_id = request.form.get("book_id")
    rating = request.form.get("rating")
    genre = request.form.get("genre", "")
    mood1 = request.form.get("mood1", "")
    mood2 = request.form.get("mood2", "")
    mood = request.form.get("mood", "")  # BARU: buat kondisi 1 (dashboard)

    source = request.form.get("source", "search")
    ref_book_id = request.form.get("ref_book_id", "")  # BARU

    if not book_id or not rating:
        flash("Rating gagal disimpan, coba lagi ya!", "error")
        if source == "my_ratings":
            return redirect(url_for("my_ratings"))
        if source == "similar_books" and ref_book_id:
            return redirect(url_for("similar_books", book_id=ref_book_id))
        if source == "dashboard":
            return redirect(url_for("home"))
        return redirect(url_for("search", genre=genre, mood1=mood1, mood2=mood2))

    existing_rating = UserRating.query.filter_by(
        user_id=session["user_id"],
        book_id=book_id
    ).first()

    if existing_rating:
        existing_rating.rating = int(rating)
    else:
        new_rating = UserRating(
            user_id=session["user_id"],
            book_id=book_id,
            rating=int(rating)
        )
        db.session.add(new_rating)

    db.session.commit()
    flash("Rating berhasil disimpan!", "success")

    # BARU: redirect sesuai source
    if source == "my_ratings":
        return redirect(url_for("my_ratings"))
    if source == "similar_books" and ref_book_id:
        return redirect(url_for("similar_books", book_id=ref_book_id))
    if source == "dashboard":
        return redirect(url_for("home"))
    return redirect(url_for("search", genre=genre, mood1=mood1, mood2=mood2))

@app.route("/search", methods=["GET", "POST"])
def search():
    books = None
    message = None

    genre = request.values.get("genre", "")
    mood1 = request.values.get("mood1", "")
    mood2 = request.values.get("mood2", "")

    if genre and mood1:
        mood_query = f"{mood1} {mood2}" if mood2 else mood1

        books = recommend_books(genre, mood_query)

        if books is None or len(books) == 0:
            message = (
                "Yahhh, belum ada nih buku dengan kombinasi genre dan mood tersebut.\n"
                "Coba pilih mood lain atau tunggu koleksi Readora bertambah yaa"
            )
        else:
            user_id = session.get("user_id")
            for b in books:
                b["UserRating"] = None
                if user_id:
                    existing = UserRating.query.filter_by(
                        user_id=user_id,
                        book_id=b["ID"]
                    ).first()
                    if existing:
                        b["UserRating"] = existing.rating
    elif request.method == "POST":
        message = "Silakan pilih Genre dan Mood terlebih dahulu!"

    return render_template(
        "recommend.html",
        books=books,
        message=message,
        selected_genre=genre,
        selected_mood1=mood1,
        selected_mood2=mood2,
        active_page="search"
    )

@app.route("/my_ratings")
def my_ratings():
    """
    Halaman riwayat rating. Menampilkan semua buku yang pernah
    dirating user dalam bentuk card, lengkap dengan rating widget
    yang sudah ke-set sesuai rating tersimpan, jadi user bisa
    langsung lihat/ubah rating tanpa harus ketik ulang judul buku.
    """
    if not session.get("user_id"):
        return redirect(url_for("login"))

    rows = (
        db.session.query(UserRating, Book)
        .join(Book, UserRating.book_id == Book.id)
        .filter(UserRating.user_id == session["user_id"])
        .order_by(UserRating.created_at.desc())
        .all()
    )

    books = []
    for ur, book in rows:
        b = _book_to_dict(book)
        b["UserRating"] = ur.rating
        books.append(b)

    message = None
    if not books:
        message = (
            "Kamu belum kasih rating ke buku manapun nih.\n"
            "Yuk cari rekomendasi buku dan kasih rating favoritmu!"
        )

    return render_template(
        "my_ratings.html",
        books=books,
        message=message,
        active_page="my_ratings"
    )

@app.route("/help")
def help_page():
    return render_template("help.html", active_page="help")

@app.route("/howtouse")
def how_to_use():
    return redirect(url_for("help_page"))

@app.route("/faq")
def faq():
    return redirect(url_for("help_page"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        
        # Cari user di database SQL
        user = User.query.filter_by(email=email).first()
        
        # Cocokkan password dengan hash yang tersimpan
        if user and check_password_hash(user.password, password):
            session["user"] = user.email
            session["user_id"] = user.id
            print("LOGIN BERHASIL")
            print(session)
            return redirect(url_for("home"))
        else:
            error = "Username atau password salah!"
    return render_template("login.html", active_page="login", error=error)

@app.route("/similar_books/<int:book_id>")
def similar_books(book_id):
    if not session.get("user_id"):
        return redirect(url_for("login"))

    user_rating = UserRating.query.filter_by(
        user_id=session["user_id"],
        book_id=book_id
    ).first()

    if not user_rating or user_rating.rating < 4:
        return render_template(
            "similar_books.html",
            books=None,
            message=(
                "Kasih rating 4 atau 5 dulu yuk ke buku ini, "
                "biar Readora bisa kasih rekomendasi yang mirip!"
            )
        )

    books = recommend_similar_books(book_id)

    if not books:
        return render_template(
            "similar_books.html",
            books=None,
            message="Belum ada rekomendasi buku serupa."
        )

    # BARU: tambahin UserRating tiap buku rekomendasi
    user_id = session["user_id"]
    for b in books:
        b["UserRating"] = None
        existing = UserRating.query.filter_by(user_id=user_id, book_id=b["ID"]).first()
        if existing:
            b["UserRating"] = existing.rating

    return render_template("similar_books.html", books=books, message=None)

@app.route("/for_you")
def for_you():
    if not session.get("user_id"):
        return redirect(url_for("login"))

    books = recommend_by_favorites(session["user_id"])

    message = None
    if not books:
        message = (
            "Belum ada rekomendasi personal nih.\n"
            "Kasih rating 4 atau 5 ke buku yang kamu suka dulu yaa, "
            "biar Readora bisa nyari yang mirip!"
        )

    return render_template(
        "for_you.html",
        books=books,
        message=message,
        active_page="for_you"
    )

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    success = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        if not email or not password or not confirm_password:
            error = "Semua kolom harus diisi!"
        elif password != confirm_password:
            error = "Konfirmasi password tidak sesuai!"
        else:
            # Periksa apakah username sudah terdaftar
            existing_user = User.query.filter_by(email=email).first()

            if existing_user:
                error = "Email sudah digunakan!"
            else:
                # Enkripsi password sebelum disimpan ke database
                hashed_pwd = generate_password_hash(password)
                new_user = User(email=email, password=hashed_pwd)
                
                db.session.add(new_user)
                db.session.commit()
                success = "Pendaftaran berhasil! Silakan masuk."
                
    return render_template("register.html", active_page="login", error=error, success=success)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)