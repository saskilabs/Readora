# migrate_data.py
import pandas as pd
from app1 import app, db, Book, User
from werkzeug.security import generate_password_hash

def migrate():
    with app.app_context():
        # 1. Buat semua tabel jika belum ada
        print("Membuat tabel database...")
        db.create_all()

        # 3. Migrasi data buku dari CSV ke SQL
        # Mengecek apakah data buku sudah ada di database
        if Book.query.first() is None:
            print("Membaca file CSV...")
            try:
                df = pd.read_csv("dataset/DATASETBUKUFINAL.csv")
                
                # Pembersihan data (sama seperti logic app.py sebelumnya)
                df["Cover"] = df["Cover"].astype(str).str.replace("covers/", "", regex=False)
                for col in ["Mood Utama", "Mood Pendukung1", "Mood Pendukung2"]:
                    df[col] = df[col].astype(str).str.strip().str.replace("Hearwarming", "Heartwarming", regex=False)
                
                print("Memasukkan data buku ke database SQL...")
                for _, row in df.iterrows():
                    book = Book(
                        title=row["Judul"],
                        author=row["Penulis"],
                        genre=row["Genre"],
                        mood_utama=row["Mood Utama"],
                        mood_pendukung1=row["Mood Pendukung1"],
                        mood_pendukung2=row["Mood Pendukung2"],
                        cover=row["Cover"],
                        sinopsis=row["Sinopsis"]
                    )
                    db.session.add(book)
                
                db.session.commit()
                print(f"Sukses memindahkan {len(df)} buku ke database SQL!")
            except Exception as e:
                print(f"Gagal melakukan migrasi buku: {e}")
        else:
            print("Data buku sudah ada di database SQL, melewati langkah migrasi buku.")

if __name__ == "__main__":
    migrate()