# 🎗️ DrawDiff — Automatické porovnání výkresů (PDF)

## 🇹🇼 Popis

**DrawDiff** je nástroj pro vizuální porovnání dvou verzí stavebního výkresu ve formátu PDF.
Z obou souborů vytvoří překryv (overlay), kde jsou změny barevně odlišené:

* **Zeleně** = původní čáry
* **Červeně** = nové prvky

Lze spustit lokálně nebo provozovat jako trvalou FastAPI službu na Windows serveru.

---

## ⚙️ Požadavky

* **Python 3.12 nebo nověší**
* **PIP** (správce balíčků)
* **Windows 10/11 nebo Server 2019+**
* Práva pro zápis do složky `work/`

---

## 🚀 Instalace na nový stroj

### 1. Stažení a příprava

```powershell
git clone https://github.com/<tvoje-repozitare>/DrawDiff.git
cd DrawDiff/server
```

### 2. Vytvoření virtuálního prostředí

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalace knihoven

```powershell
pip install -r requirements.txt
```

### 4. Spuštění serveru

```powershell
.\venv\Scripts\uvicorn.exe server:app --host 0.0.0.0 --port 8000
```

### 5. Otevření API dokumentace

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧩 Použití

1. Klikni na **Authorize** a vlož API klíč z `.env`.
2. Vyber endpoint `/drawdiff`.
3. Nahraj dvě PDF (původní a novou verzi).
4. Vyber variantu:

   * `default` – automatické zarovnání a ořez
   * `fixed` – jednoduché položení na 3×3 plátno (bez zarovnání)
5. Klikni **Execute** → v adresáři `work/<job_id>/` se vytvoří výsledek.

---

## 🧠 Spuštění jako Windows služba (volitelné)

1. Stáhni [NSSM](https://nssm.cc/download) a rozbal do `DrawDiff/deploy/nssm/`
2. V PowerShellu (spusť jako admin):

```powershell
& ".\deploy\nssm\nssm.exe" install DrawDiff `
  "D:\Source\python\DrawDiff\server\venv\Scripts\python.exe" `
  "-m uvicorn server:app --host 0.0.0.0 --port 8000"
& ".\deploy\nssm\nssm.exe" set DrawDiff AppDirectory "D:\Source\python\DrawDiff\server"
& ".\deploy\nssm\nssm.exe" set DrawDiff Start SERVICE_AUTO_START
& ".\deploy\nssm\nssm.exe" start DrawDiff
```

3. Správa služby:

```powershell
Stop-Service DrawDiff
Start-Service DrawDiff
```

---

## 🔒 Konfigurace (.env)

```ini
WORK_DIR=../work
CORS_ORIGINS=http://127.0.0.1,http://localhost
API_KEY=secret_demo_key
```

---

## 📁 Struktura výsledků

```
work/2025-11-27_19-05-44_ab12cd34/
│── old.pdf
│── new.pdf
│── overlay.png
│── debug_fixed_canvas.png
```

---

## 🛋️ Automatické čištění složky `work`

```powershell
Get-ChildItem .\work -Directory |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
  Remove-Item -Recurse -Force
```

---

# 🇬🇧 ENGLISH VERSION

## 📊 Overview

**DrawDiff** is a tool for visual comparison of two construction drawings in PDF format.
It creates a color overlay highlighting differences between drawings:

* **Green** = original drawing
* **Red** = new elements

It can run locally or as a persistent FastAPI service on Windows Server.

---

## ⚙️ Requirements

* **Python 3.12 or newer**
* **PIP** package manager
* **Windows 10/11 or Server 2019+**
* Write access to the `work/` folder

---

## 🚀 Installation on a new machine

### 1. Clone repository

```bash
git clone https://github.com/<your-repo>/DrawDiff.git
cd DrawDiff/server
```

### 2. Create virtual environment

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run server

```bash
.\venv\Scripts\uvicorn.exe server:app --host 0.0.0.0 --port 8000
```

### 5. Open API documentation

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔄 Usage

1. Click **Authorize** and enter the API key from `.env`.
2. Select `/drawdiff` endpoint.
3. Upload two PDF files (old and new version).
4. Choose variant:

   * `default` – automatic alignment and crop
   * `fixed` – simple 3x3 layout without alignment
5. Click **Execute** → results will appear in `work/<job_id>/`.

---

## 🧠 Run as Windows service (optional)

1. Install [NSSM](https://nssm.cc/download) and extract to `DrawDiff/deploy/nssm/`.
2. In PowerShell (Run as Administrator):

```powershell
& ".\deploy\nssm\nssm.exe" install DrawDiff `
  "D:\Source\python\DrawDiff\server\venv\Scripts\python.exe" `
  "-m uvicorn server:app --host 0.0.0.0 --port 8000"
& ".\deploy\nssm\nssm.exe" set DrawDiff AppDirectory "D:\Source\python\DrawDiff\server"
& ".\deploy\nssm\nssm.exe" set DrawDiff Start SERVICE_AUTO_START
& ".\deploy\nssm\nssm.exe" start DrawDiff
```

3. Manage service:

```powershell
Stop-Service DrawDiff
Start-Service DrawDiff
```

---

## 🔒 Configuration (.env)

```ini
WORK_DIR=../work
CORS_ORIGINS=http://127.0.0.1,http://localhost
API_KEY=secret_demo_key
```

---

## 📁 Output structure

```
work/2025-11-27_19-05-44_ab12cd34/
│── old.pdf
│── new.pdf
│── overlay.png
│── debug_fixed_canvas.png
```

---

## 🛋️ Auto-clean `work` folder

```powershell
Get-ChildItem .\work -Directory |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
  Remove-Item -Recurse -Force
```

---

💬 **Note:** Project runs fully offline and uses only Python libraries. It can be safely deployed in company environments without internet access.
