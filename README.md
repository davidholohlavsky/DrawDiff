# 🏗️ DrawDiff Prototype

Automatizované porovnání stavebních výkresů v PDF — **vizuální zvýraznění rozdílů** mezi dvěma verzemi dokumentu.

---

## 🚀 Cíl projektu
- Načíst dvě PDF (původní a novou verzi výkresu)  
- Vyhodnotit pixely, kde se výkresy liší  
- Vygenerovat **overlay** obrázek:
  - původní vrstva = **zelená (nahoře)**  
  - nová vrstva = **červená (dole, přepisuje změny)**  

---

## 📂 Struktura projektu
```
DrawDiff/
├── server/
│   ├── drawdiff.py         ← logika porovnávání
│   ├── server.py           ← FastAPI server
│   ├── requirements.txt    ← potřebné knihovny
│   ├── .env                ← lokální konfigurace
│   └── .env.example
│
├── work/                   ← automaticky vznikající složka s výsledky
│   └── job_YYYY-MM-DD_...  ← každé porovnání má vlastní podsložku
│
└── deploy/                 ← (volitelně) skripty pro instalaci jako služba
```

---

## ⚙️ Požadavky
- **Python 3.12.x** (doporučeno 3.12.9)
- PowerShell 7+
- Knihovny z `requirements.txt` (viz níže)
- Přístupová práva pro zápis do složky `work/`

---

## 🧩 První spuštění (lokálně)

### 1️⃣ Aktivuj virtuální prostředí
```powershell
cd D:\Source\python\DrawDiff\server
.\venv\Scripts\Activate.ps1
```

### 2️⃣ Spusť server
```powershell
.\venv\Scripts\uvicorn.exe server:app --host 0.0.0.0 --port 8000
```

### 3️⃣ Otevři dokumentaci v prohlížeči
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

- Klikni na **Authorize** (zámek)
- Do pole vlož API klíč z `.env`  
  ```
  secret_demo_key
  ```
- Pak spusť `/drawdiff` → *Try it out* → nahraj 2 PDF → *Execute*  
- Výsledek `overlay.png` se vytvoří ve složce:
  ```
  D:\Source\python\DrawDiff\work\job_YYYY-MM-DD_HH-MM-SS_xxxx
  ```

---

🔧 Instalace na nový stroj
1️⃣ Nainstaluj Python 3.12.x

Stáhneš z oficiálního webu, stačí jedna instalace.

2️⃣ Vytvoř nové virtuální prostředí
python -m venv venv
3️⃣ Aktivuj prostředí
.\venv\Scripts\Activate.ps1
4️⃣ Nainstaluj knihovny
pip install -r requirements.txt

Hotovo — systém je připravený ke spuštění.

## 🧠 Spuštění jako Windows služba (volitelné)

### 1️⃣ Nainstaluj [NSSM – Non-Sucking Service Manager](https://nssm.cc/download)

Rozbal `nssm.exe` například do:
```
D:\Source\python\DrawDiff\deploy\nssm\
```

### 2️⃣ Zaregistruj službu (PowerShell, spouštěj jako admin)
```powershell
cd D:\Source\python\DrawDiff\deploy
$nssm = ".\nssm\nssm.exe"
& $nssm install DrawDiff `
  "D:\Source\python\DrawDiff\server\venv\Scripts\python.exe" `
  "-m uvicorn server:app --host 0.0.0.0 --port 8000"
& $nssm set DrawDiff AppDirectory "D:\Source\python\DrawDiff\server"
& $nssm set DrawDiff Start SERVICE_AUTO_START
& $nssm start DrawDiff
```

### 3️⃣ Kontrola služby
```powershell
Get-Service DrawDiff
```

### 4️⃣ Logy (volitelně)
```powershell
mkdir D:\Source\python\DrawDiff\logs
& $nssm set DrawDiff AppStdout "D:\Source\python\DrawDiff\logs\drawdiff.out.log"
& $nssm set DrawDiff AppStderr "D:\Source\python\DrawDiff\logs\drawdiff.err.log"
```

---

## 🔒 Soubor `.env`
```
WORK_DIR=../work
CORS_ORIGINS=http://127.0.0.1,http://localhost
API_KEY=secret_demo_key
```

---

## 📸 Výsledek porovnání
- Původní (první) PDF: **zelený tón (nahoře)**  
- Nové (druhé) PDF: **červený tón (dole)**  
- Překryvy → červená přepíše zelenou, zobrazí změnu.

---

## 🧹 Údržba
Staré výsledky můžeš mazat ručně, nebo automaticky např. PowerShellem:
```powershell
Get-ChildItem D:\Source\python\DrawDiff\work -Directory |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
  Remove-Item -Recurse -Force
```

---

## ✅ Shrnutí příkazů pro firemní notebook

| Krok | Příkaz | Poznámka |
|------|---------|----------|
| Aktivace prostředí | `.\venv\Scripts\Activate.ps1` | ve složce `server` |
| Spuštění serveru | `.\venv\Scripts\uvicorn.exe server:app --port 8000` | Python 3.12 |
| Otevření API | `http://127.0.0.1:8000/docs` | v prohlížeči |
| Instalace služby | viz PowerShell skript výše | NSSM nutné |
| Stop služby | `Stop-Service DrawDiff` | |
| Start služby | `Start-Service DrawDiff` | |

---

💬 **Poznámka pro IT prezentaci:**  
Projekt běží plně lokálně, bez připojení k internetu.  
Používá jen Python knihovny a generuje vizuální porovnání PDF.  
Po schválení se může služba přesunout na firemní server jako trvalá interní API služba.
