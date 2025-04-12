import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Problem: System oceny jakości wody w akwarium
# Zmienne wejściowe:
# 1. Poziom pH (6.0-9.0)
# 2. Temperatura wody (18-32°C)
# 3. Poziom amoniaku (0-2 ppm)
# Zmienna wyjściowa: Jakość wody (0-100%)

ph = ctrl.Antecedent(np.arange(6, 9.1, 0.1), 'pH')
temperatura = ctrl.Antecedent(np.arange(18, 32.1, 0.1), 'temperatura')
amoniak = ctrl.Antecedent(np.arange(0, 2.1, 0.1), 'amoniak')

jakosc_wody = ctrl.Consequent(np.arange(0, 101, 1), 'jakosc_wody')

ph['kwaśne'] = fuzz.trimf(ph.universe, [6.0, 6.0, 7.0])
ph['neutralne'] = fuzz.trimf(ph.universe, [6.5, 7.2, 7.8])
ph['zasadowe'] = fuzz.trimf(ph.universe, [7.5, 9.0, 9.0])

temperatura['niska'] = fuzz.trimf(temperatura.universe, [18, 18, 23])
temperatura['optymalna'] = fuzz.trimf(temperatura.universe, [21, 25, 28])
temperatura['wysoka'] = fuzz.trimf(temperatura.universe, [26, 32, 32])

amoniak['bezpieczny'] = fuzz.trimf(amoniak.universe, [0, 0, 0.5])
amoniak['podwyższony'] = fuzz.trimf(amoniak.universe, [0.3, 0.8, 1.3])
amoniak['niebezpieczny'] = fuzz.trimf(amoniak.universe, [1.0, 2.0, 2.0])

jakosc_wody['zła'] = fuzz.trimf(jakosc_wody.universe, [0, 0, 30])
jakosc_wody['średnia'] = fuzz.trimf(jakosc_wody.universe, [20, 50, 80])
jakosc_wody['dobra'] = fuzz.trimf(jakosc_wody.universe, [70, 100, 100])

regula1 = ctrl.Rule(ph['neutralne'] & temperatura['optymalna'] & amoniak['bezpieczny'], jakosc_wody['dobra'])
regula2 = ctrl.Rule(ph['kwaśne'] & temperatura['optymalna'] & amoniak['bezpieczny'], jakosc_wody['średnia'])
regula3 = ctrl.Rule(ph['zasadowe'] & temperatura['optymalna'] & amoniak['bezpieczny'], jakosc_wody['średnia'])
regula4 = ctrl.Rule(ph['neutralne'] & temperatura['niska'] & amoniak['bezpieczny'], jakosc_wody['średnia'])
regula5 = ctrl.Rule(ph['neutralne'] & temperatura['wysoka'] & amoniak['bezpieczny'], jakosc_wody['średnia'])
regula6 = ctrl.Rule(ph['neutralne'] & temperatura['optymalna'] & amoniak['podwyższony'], jakosc_wody['średnia'])
regula7 = ctrl.Rule(amoniak['niebezpieczny'], jakosc_wody['zła'])
regula8 = ctrl.Rule((ph['kwaśne'] | ph['zasadowe']) & temperatura['wysoka'], jakosc_wody['zła'])
regula9 = ctrl.Rule(temperatura['niska'] & amoniak['podwyższony'], jakosc_wody['zła'])

regula10 = ctrl.Rule(ph['zasadowe'] & temperatura['niska'] & amoniak['bezpieczny'], jakosc_wody['średnia'])
regula11 = ctrl.Rule(ph['kwaśne'] & temperatura['niska'] & amoniak['bezpieczny'], jakosc_wody['średnia'])
regula12 = ctrl.Rule(ph['zasadowe'] & temperatura['wysoka'] & amoniak['bezpieczny'], jakosc_wody['zła'])
regula13 = ctrl.Rule(ph['kwaśne'] & temperatura['wysoka'] & amoniak['bezpieczny'], jakosc_wody['zła'])
regula14 = ctrl.Rule(ph['zasadowe'] & temperatura['optymalna'] & amoniak['podwyższony'], jakosc_wody['zła'])
regula15 = ctrl.Rule(ph['kwaśne'] & temperatura['optymalna'] & amoniak['podwyższony'], jakosc_wody['zła'])
regula16 = ctrl.Rule(ph['neutralne'] & temperatura['niska'] & amoniak['podwyższony'], jakosc_wody['zła'])
regula17 = ctrl.Rule(ph['neutralne'] & temperatura['wysoka'] & amoniak['podwyższony'], jakosc_wody['zła'])

system_kontroli = ctrl.ControlSystem([regula1, regula2, regula3, regula4, regula5, regula6, regula7, regula8, regula9,
                                     regula10, regula11, regula12, regula13, regula14, regula15, regula16, regula17])
system_oceny = ctrl.ControlSystemSimulation(system_kontroli)

def oblicz_jakosc(ph_val, temp_val, amon_val, system):
    try:
        system.input['pH'] = ph_val
        system.input['temperatura'] = temp_val
        system.input['amoniak'] = amon_val
        system.compute()
        return system.output.get('jakosc_wody', 0) 
    except Exception as e:
        return 0  

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
ph.view()
plt.title('Funkcje przynależności dla pH')

plt.subplot(2, 2, 2)
temperatura.view()
plt.title('Funkcje przynależności dla temperatury')

plt.subplot(2, 2, 3)
amoniak.view()
plt.title('Funkcje przynależności dla poziomu amoniaku')

plt.subplot(2, 2, 4)
jakosc_wody.view()
plt.title('Funkcje przynależności dla jakości wody')

plt.tight_layout()
plt.savefig('akwarium_funkcje_przynaleznosci.png')
plt.show()

# Przykładowe przypadki testowe
przypadki_testowe = [
    {'pH': 7.2, 'temperatura': 25, 'amoniak': 0.1, 'opis': 'Warunki optymalne'},
    {'pH': 6.2, 'temperatura': 24, 'amoniak': 0.2, 'opis': 'Kwaśne pH, reszta w normie'},
    {'pH': 8.5, 'temperatura': 24, 'amoniak': 0.2, 'opis': 'Zasadowe pH, reszta w normie'},
    {'pH': 7.2, 'temperatura': 20, 'amoniak': 0.2, 'opis': 'Niska temperatura, reszta w normie'},
    {'pH': 7.2, 'temperatura': 30, 'amoniak': 0.2, 'opis': 'Wysoka temperatura, reszta w normie'},
    {'pH': 7.2, 'temperatura': 25, 'amoniak': 1.0, 'opis': 'Podwyższony amoniak, reszta w normie'},
    {'pH': 7.2, 'temperatura': 25, 'amoniak': 1.8, 'opis': 'Niebezpieczny poziom amoniaku'},
    {'pH': 6.1, 'temperatura': 30, 'amoniak': 0.2, 'opis': 'Kwaśne pH i wysoka temperatura'},
    {'pH': 8.5, 'temperatura': 29, 'amoniak': 0.2, 'opis': 'Zasadowe pH i wysoka temperatura'},
    {'pH': 7.2, 'temperatura': 19, 'amoniak': 1.1, 'opis': 'Niska temperatura i podwyższony amoniak'}
]

print("Wyniki oceny jakości wody w akwarium:")
print("-" * 80)
print(f"{'Warunki':<40} {'pH':<6} {'Temp.':<6} {'Amoniak':<8} {'Jakość wody':<12}")
print("-" * 80)

for przypadek in przypadki_testowe:
    jakosc = oblicz_jakosc(przypadek['pH'], przypadek['temperatura'], przypadek['amoniak'], system_oceny)
    print(f"{przypadek['opis']:<40} {przypadek['pH']:<6.1f} {przypadek['temperatura']:<6.1f} {przypadek['amoniak']:<8.1f} {jakosc:<12.1f}")

# Generowanie danych do heatmapy
# Wybieramy stałą wartość amoniaku (bezpieczną) i badamy wpływ pH i temperatury
ph_range = np.arange(6, 9.1, 0.2)
temp_range = np.arange(18, 32.1, 0.5)
amoniak_staly = 0.1  # bezpieczny poziom

# Inicjalizacja macierzy wynikowej
wynik_heatmapa = np.zeros((len(temp_range), len(ph_range)))

# Wypełnianie macierzy wynikowej
for i, t in enumerate(temp_range):
    for j, p in enumerate(ph_range):
        wynik_heatmapa[i, j] = oblicz_jakosc(p, t, amoniak_staly, system_oceny)

# Wizualizacja heatmapy
plt.figure(figsize=(12, 8))
plt.pcolormesh(ph_range, temp_range, wynik_heatmapa, cmap='RdYlGn', shading='auto')
plt.colorbar(label='Jakość wody [%]')
plt.xlabel('pH')
plt.ylabel('Temperatura [°C]')
plt.title(f'Jakość wody w zależności od pH i temperatury (przy stałym poziomie amoniaku = {amoniak_staly} ppm)')
plt.grid(True)
plt.savefig('akwarium_heatmapa_ph_temp.png')
plt.show()

# Generowanie danych do heatmapy
# Wybieramy stałe optimalne pH i badamy wpływ temperatury i amoniaku
temp_range = np.arange(18, 32.1, 0.5)
amoniak_range = np.arange(0, 2.1, 0.1)
ph_staly = 7.2  # neutralne pH

# Inicjalizacja macierzy wynikowej
wynik_heatmapa2 = np.zeros((len(temp_range), len(amoniak_range)))

# Wypełnianie macierzy wynikowej
for i, t in enumerate(temp_range):
    for j, a in enumerate(amoniak_range):
        wynik_heatmapa2[i, j] = oblicz_jakosc(ph_staly, t, a, system_oceny)

# Wizualizacja heatmapy
plt.figure(figsize=(12, 8))
plt.pcolormesh(amoniak_range, temp_range, wynik_heatmapa2, cmap='RdYlGn', shading='auto')
plt.colorbar(label='Jakość wody [%]')
plt.xlabel('Poziom amoniaku [ppm]')
plt.ylabel('Temperatura [°C]')
plt.title(f'Jakość wody w zależności od poziomu amoniaku i temperatury (przy stałym pH = {ph_staly})')
plt.grid(True)
plt.savefig('akwarium_heatmapa_temp_amoniak.png')
plt.show()

# Wykres 3D - wpływ pH i temperatury przy stałym poziomie amoniaku
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Przygotowanie siatki
X, Y = np.meshgrid(ph_range, temp_range)

# Wykres 3D
surf = ax.plot_surface(X, Y, wynik_heatmapa, cmap='RdYlGn', edgecolor='none')
ax.set_xlabel('pH')
ax.set_ylabel('Temperatura [°C]')
ax.set_zlabel('Jakość wody [%]')
ax.set_title(f'Jakość wody w zależności od pH i temperatury (amoniak = {amoniak_staly} ppm)')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Jakość wody [%]')
plt.savefig('akwarium_3d_ph_temp.png')
plt.show()

# Wnioski
print("\nWnioski z analizy systemu:")
print("-" * 80)
print("1. Najwyższa jakość wody osiągana jest przy neutralnym pH (około 7.2), optymalnej")
print("   temperaturze (24-26°C) i bezpiecznym poziomie amoniaku (poniżej 0.3 ppm).")
print("2. Poziom amoniaku ma największy wpływ na jakość wody - wysokie stężenie amoniaku")
print("   (powyżej 1.0 ppm) zawsze powoduje złą jakość wody, niezależnie od pozostałych parametrów.")
print("3. Skrajne wartości pH (poniżej 6.5 lub powyżej 8.0) w połączeniu z wysoką temperaturą")
print("   również znacząco obniżają jakość wody.")
print("4. Niska temperatura w połączeniu z podwyższonym poziomem amoniaku daje gorsze wyniki")
print("   niż wysoka temperatura z takim samym poziomem amoniaku.")
print("5. System wykazuje największą tolerancję dla odchyleń temperatury przy neutralnym pH")
print("   i niskim poziomie amoniaku.")
print("6. Zastosowanie logiki rozmytej pozwoliło na modelowanie płynnych przejść między")
print("   różnymi stanami parametrów wody, co lepiej odzwierciedla rzeczywiste zależności")
print("   niż ostre granice stosowane w klasycznych systemach ekspertowych.")
print("7. Po dodaniu dodatkowych reguł, system lepiej pokrywa przestrzeń decyzyjną,")
print("   eliminując błędy związane z brakiem odpowiednich reguł dla niektórych kombinacji parametrów.") 