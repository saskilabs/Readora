from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from datetime import datetime, date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "readora_cozy_secret_key_123"

# Database Configuration (MySQL Workbench)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/readora_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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
    genre = db.Column(db.String(100))
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
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    book_id = db.Column(db.Integer, db.ForeignKey('books.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

class DailyMood(db.Model):
    __tablename__ = "daily_moods"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer,db.ForeignKey("users.id"),nullable=False)
    mood = db.Column(db.String(100), nullable=False)
    selected_date = db.Column(db.Date,nullable=False)

with app.app_context():
    db.create_all()


def recommend_books(genre_input, mood_input, top_n=5):
    # Query database SQL untuk mendapatkan buku dengan genre terkait
    books_query = Book.query.filter_by(genre=genre_input).all()
    if not books_query:
        return None

    # Ubah hasil query SQLAlchemy menjadi DataFrame untuk diproses oleh cosine_similarity
    data_list = []
    for b in books_query:
        data_list.append({
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
            # Buat representasi content untuk TF-IDF
            "content": (
                (str(b.mood_utama) + " ") * 5 +
                (str(b.mood_pendukung1) + " ") * 2 +
                str(b.mood_pendukung2)
            )
        })
    
    filtered_df = pd.DataFrame(data_list)

    mood_list = mood_input.split()

    # Validasi apakah mood yang dicari ada di dalam data genre ini
    mood_exists =any(
            (filtered_df["Mood Utama"] == mood).any() or
            (filtered_df["Mood Pendukung1"] == mood).any() or
            (filtered_df["Mood Pendukung2"] == mood).any()
            for mood in mood_list
    )

    if not mood_exists:
        return None

    # Lakukan perhitungan Cosine Similarity (TF-IDF) seperti biasa
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(filtered_df["content"])
    query_vector = tfidf.transform([mood_input])

    similarities = cosine_similarity(query_vector, tfidf_matrix)
    scores = similarities.flatten()

    valid_indices = [i for i, score in enumerate(scores) if score > 0]
    if len(valid_indices) == 0:
        return None

    top_indices = sorted(
        valid_indices,
        key=lambda x: scores[x],
        reverse=True
    )[:top_n]

    recommendations = filtered_df.iloc[top_indices]
    return recommendations.to_dict(orient="records")

def recommend_similar_books(book_id, top_n=5):

    # Ambil buku yang diberi rating user
    selected_book = Book.query.get(book_id)

    if not selected_book:
        return None

    # Ambil genre buku
    genre = selected_book.genre

    # Gabungkan mood buku
    mood_query = " ".join(filter(None, [
        selected_book.mood_utama,
        selected_book.mood_pendukung1,
        selected_book.mood_pendukung2
    ]))

    # Ambil semua buku dengan genre yang sama,
    # kecuali buku yang sedang dipilih
    books_query = Book.query.filter(
        Book.genre == genre,
        Book.id != book_id
    ).all()

    if not books_query:
        return None

    data_list = []

    for b in books_query:

        data_list.append({
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
            "content": (
                (str(b.mood_utama) + " ") * 5 +
                (str(b.mood_pendukung1) + " ") * 2 +
                str(b.mood_pendukung2)
            )
        })

    filtered_df = pd.DataFrame(data_list)

        # Hitung TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(filtered_df["content"])

    # Query berasal dari mood buku yang dipilih user
    query_vector = tfidf.transform([mood_query])

    similarities = cosine_similarity(query_vector, tfidf_matrix)
    scores = similarities.flatten()

    valid_indices = [i for i, score in enumerate(scores) if score > 0]

    if len(valid_indices) == 0:
        return None

    top_indices = sorted(
        valid_indices,
        key=lambda x: scores[x],
        reverse=True
    )[:top_n]

    recommendations = filtered_df.iloc[top_indices]

    return recommendations.to_dict(orient="records")


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

    # Semua buku yang pernah dirating (apapun nilainya) -> exclude dari rekomendasi
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
        mood_query = " ".join(filter(None, [
            fav_book.mood_utama,
            fav_book.mood_pendukung1,
            fav_book.mood_pendukung2
        ]))

        candidates = Book.query.filter(
            Book.genre == genre,
            ~Book.id.in_(rated_book_ids)
        ).all()

        if not candidates:
            continue

        contents = []
        for b in candidates:
            contents.append(
                (str(b.mood_utama) + " ") * 5 +
                (str(b.mood_pendukung1) + " ") * 2 +
                str(b.mood_pendukung2)
            )

        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(contents)
        query_vector = tfidf.transform([mood_query])

        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        for i, score in enumerate(similarities):
            if score <= 0:
                continue
            b = candidates[i]
            # Buku yang kamu kasih rating 5 punya bobot lebih besar dari rating 4
            weighted_score = score * fav.rating
            score_map[b.id] = score_map.get(b.id, 0) + weighted_score
            book_cache[b.id] = b

    if not score_map:
        return None

    top_ids = sorted(score_map, key=lambda k: score_map[k], reverse=True)[:top_n]

    results = []
    for bid in top_ids:
        b = book_cache[bid]
        results.append({
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
        })

    return results


@app.route("/")
def home():
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
            books = recommend_books(
                genre, selected_mood, top_n=2
            )
            if books:
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

    if not book_id or not rating:
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
    return redirect(url_for("search", genre=genre, mood1=mood1, mood2=mood2))

@app.route("/search", methods=["GET", "POST"])
def search():
    books = None
    message = None

    genre = request.values.get("genre", "")
    mood1 = request.values.get("mood1", "")
    mood2 = request.values.get("mood2", "")

    # Jalanin pencarian kalau ada genre & mood1 terisi,
    # baik dari submit form (POST) maupun redirect abis rating (GET)
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

    return render_template(
        "similar_books.html",
        books=books,
        message=None
    )

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
    app.run(debug=True, port=5002)