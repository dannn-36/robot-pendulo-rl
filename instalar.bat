@echo off
rem Instala (o repara) el entorno de Python del simulador y comprueba el ordenador.
rem Doble clic en este archivo, o desde una terminal en esta carpeta: instalar.bat
setlocal
cd /d "%~dp0"

rem --- 1. Un .venv copiado de otro ordenador apunta a un Python que aqui no existe: se rehace.
if not exist ".venv\Scripts\python.exe" goto buscar_python
".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if not errorlevel 1 goto instalar
echo El entorno .venv no funciona en este ordenador (seguramente se copio de otro PC). Se vuelve a crear...
rmdir /s /q ".venv"

:buscar_python
rem --- 2. Python compatible: 3.10 a 3.14, preferiblemente el mas nuevo.
set "PY="
for %%v in (3.14 3.13 3.12 3.11 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if defined PY goto crear
python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>&1 && set "PY=python"
if defined PY goto crear
echo.
echo No se encontro Python 3.10 - 3.14 en este ordenador.
echo Instala Python 3.13 desde https://www.python.org/downloads/ marcando "Add python.exe to PATH"
echo y vuelve a ejecutar instalar.bat
goto fin_error

:crear
echo Creando el entorno .venv con %PY% ...
%PY% -m venv .venv
if errorlevel 1 goto fin_error

:instalar
rem --- 3. Con una ruta de carpeta muy larga, PyTorch no cabe en el limite de 260 caracteres de Windows.
".venv\Scripts\python.exe" diagnostico.py --comprobar-ruta
if errorlevel 1 goto fin_error

rem --- 4. Librerias. En Windows, PyTorch de PyPI es la version solo-CPU: basta para la demo.
echo Instalando librerias (la primera vez tarda unos minutos: PyTorch pesa unos 200 MB)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fallo_pip

rem --- 5. Diagnostico: grafica, modelos, velocidad y calidad recomendada.
echo.
".venv\Scripts\python.exe" diagnostico.py
echo.
echo Para usarlo, abre una terminal en esta carpeta y ejecuta:
echo    .venv\Scripts\activate
echo    python evaluar.py --pd --terreno colinas
pause
exit /b 0

:fallo_pip
echo.
echo Fallo la instalacion de librerias: lee el error de arriba (lo mas comun: sin internet).
echo Arreglalo y vuelve a ejecutar instalar.bat
:fin_error
pause
exit /b 1
