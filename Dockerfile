# Menggunakan image python sebagai base image
FROM python:3.11-slim

# Menetapkan direktori kerja di container
WORKDIR /app

# Menyalin file requirements.txt ke direktori kerja
COPY requirements.txt .

# Menginstal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin semua file ke direktori kerja
COPY . .

# Tentukan port yang akan digunakan
EXPOSE 5000

# Menjalankan aplikasi Flask
CMD ["flask", "--app", "app", "run", "--host=0.0.0.0"]
