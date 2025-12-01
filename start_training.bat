@echo off
echo ============================================================
echo    DETECTOR DE SOTAQUE BRASILEIRO - TREINAMENTO
echo ============================================================
echo.
echo Iniciando treinamento...
echo.

python -u main.py train --model cnn --epochs 30 --workers 0 --batch-size 8

echo.
echo ============================================================
echo Treinamento finalizado!
echo ============================================================
echo.
pause


