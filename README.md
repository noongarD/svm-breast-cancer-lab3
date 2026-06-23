# SVM - Breast Cancer Wisconsin (Laboratorium 3)

Projekt laboratoryjny z przedmiotu **Wstęp do sztucznej inteligencji**.
Skrypt porównuje klasyfikatory SVC dla jąder `linear` i `rbf`, różnych
wartości parametru `C` oraz danych ze standaryzacją i bez niej.

## Uruchomienie

```bash
python -m pip install -r requirements.txt
python svm_breast_cancer_lab3.py
```

Wyniki zostaną zapisane w katalogu `svm_outputs`.

## Zawartość

- `svm_breast_cancer_lab3.py` - kod eksperymentu,
- `raport_svm_lab3.pdf` - raport laboratoryjny,
- `svm_outputs/accuracy_results.csv` - tabela dokładności,
- `svm_outputs/*.png` - wykres dokładności, macierze konfuzji i wizualizacje 2D.

## Zakres eksperymentu

- zbiór Breast Cancer Wisconsin z `scikit-learn`,
- podział danych 70/30 ze stratyfikacją,
- standaryzacja dopasowana wyłącznie do zbioru uczącego,
- jądra `linear` i `rbf`,
- wartości `C`: `0.01`, `0.1`, `1`, `10`, `100`,
- accuracy i macierze konfuzji dla najlepszych modeli.
