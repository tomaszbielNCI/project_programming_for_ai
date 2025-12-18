# -*- coding: utf-8 -*-
import sys
import io
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Ustawienie kodowania wyjścia na UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Wczytaj dane
df_raw = pd.read_csv('../../temp_data/temperature_raw.csv')
df_clean = pd.read_csv('../../temp_data/exports/development_analysis/temperature_clean.csv')

# Wyciągnij kolumnę z anomaliami temperaturowymi
raw_data = df_raw['temp_anomaly'].dropna().values
clean_data = df_clean['temp_anomaly'].dropna().values

# Sprawdź normalność danych
def check_normality(data, name):
    stat, p = stats.shapiro(data)
    print(f"\nTest Shapiro-Wilka dla {name}:")
    print(f"Statystyka = {stat:.4f}, p = {p:.4f}")
    if p > 0.05:
        print(f"{name}: Dane mają rozkład normalny (p > 0.05)")
        return True
    else:
        print(f"{name}: Dane NIE mają rozkładu normalnego (p ≤ 0.05)")
        return False

# Sprawdź normalność dla obu próbek
is_raw_normal = check_normality(raw_data, 'danych surowych')
is_clean_normal = check_normality(clean_data, 'danych oczyszczonych')

# Wybierz odpowiedni test
if is_raw_normal and is_clean_normal:
    print("\nObie próbki mają rozkład normalny - wykonuję test t-Studenta")
    # Sprawdzenie równości wariancji
    levene_test = stats.levene(raw_data, clean_data)
    print(f"\nTest Levene'a dla równości wariancji: p = {levene_test.pvalue:.4f}")
    
    if levene_test.pvalue > 0.05:
        print("Wariancje są równe (p > 0.05) - używam testu t z równymi wariancjami")
        t_stat, p_value = stats.ttest_ind(raw_data, clean_data, equal_var=True)
    else:
        print("Wariancje nie są równe (p ≤ 0.05) - używam testu t Welcha (niesymetryczne wariancje)")
        t_stat, p_value = stats.ttest_ind(raw_data, clean_data, equal_var=False)
    
    test_used = "test t-Studenta"
    print(f"\nWynik {test_used}:")
    print(f"t = {t_stat:.4f}, p = {p_value:.4f}")
else:
    print("\nPrzynajmniej jedna z próbek nie ma rozkładu normalnego - wykonuję test Manna-Whitneya")
    u_stat, p_value = stats.mannwhitneyu(raw_data, clean_data, alternative='two-sided')
    test_used = "test Manna-Whitneya"
    print(f"\nStatystyka U = {u_stat:.4f}")
    print(f"\nWynik {test_used}:")
    print(f"p = {p_value:.4f}")

# Interpretacja
alpha = 0.05
if p_value < alpha:
    print("\nRóżnice są statystycznie istotne (p < 0.05)")
    print("Średnie temperatury różnią się istotnie między danymi surowymi a oczyszczonymi")
else:
    print("\nBrak istotnych statystycznie różnic (p ≥ 0.05)")
    print("Średnie temperatury nie różnią się istotnie między danymi surowymi a oczyszczonymi")

# Dodatkowa analiza - rozkłady
print("\nŚrednie wartości:")
print(f"Dane surowe: {df_raw['temp_anomaly'].mean():.4f} ± {df_raw['temp_anomaly'].std():.4f}")
print(f"Dane oczyszczone: {df_clean['temp_anomaly'].mean():.4f} ± {df_clean['temp_anomaly'].std():.4f}")

# Test normalności (dla każdej grupy osobno)
_, p_raw = stats.normaltest(df_raw['temp_anomaly'].dropna())
_, p_clean = stats.normaltest(df_clean['temp_anomaly'].dropna())

print("\nTest normalności (H0: dane mają rozkład normalny):")
print(f"Dane surowe: p = {p_raw:.4f} - {'normalne' if p_raw > 0.05 else 'nie normalne'}")
print(f"Dane oczyszczone: p = {p_clean:.4f} - {'normalne' if p_clean > 0.05 else 'nie normalne'}")

# Jeśli dane nie są normalne, możemy użyć testu nieparametrycznego
if p_raw < 0.05 or p_clean < 0.05:
    print("\nUwaga: Ponieważ dane nie mają rozkładu normalnego, rozważ użycie testu nieparametrycznego:")
    u_stat, p_mannwhitney = stats.mannwhitneyu(df_raw['temp_anomaly'].dropna(),
                                            df_clean['temp_anomaly'].dropna())
    print(f"\nTest Manna-Whitneya (dane nieparametryczne):")
    print(f"U = {u_stat:.4f}, p = {p_mannwhitney:.4f}")
    if p_mannwhitney < 0.05:
        print("Różnice są statystycznie istotne (p < 0.05)")
    else:
        print("Brak istotnych statystycznie różnic (p ≥ 0.05)")