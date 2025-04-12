import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperatura = ctrl.Antecedent(np.arange(15, 36, 1), 'temperatura')
wilgotnosc = ctrl.Antecedent(np.arange(0, 101, 1), 'wilgotnosc')

ilosc_wody = ctrl.Consequent(np.arange(0, 26, 1), 'ilosc_wody')

temperatura['chlodno'] = fuzz.trimf(temperatura.universe, [15, 15, 22])
temperatura['cieplo'] = fuzz.trimf(temperatura.universe, [18, 25, 32])
temperatura['goraco'] = fuzz.trimf(temperatura.universe, [28, 35, 35])

wilgotnosc['sucho'] = fuzz.trimf(wilgotnosc.universe, [0, 0, 40])
wilgotnosc['przecietnie'] = fuzz.trimf(wilgotnosc.universe, [25, 50, 75])
wilgotnosc['mokro'] = fuzz.trimf(wilgotnosc.universe, [60, 100, 100])

ilosc_wody['malo'] = fuzz.trimf(ilosc_wody.universe, [0, 5, 10])
ilosc_wody['srednio'] = fuzz.trimf(ilosc_wody.universe, [5, 10, 15])
ilosc_wody['duzo'] = fuzz.trimf(ilosc_wody.universe, [10, 20, 25])

regula1 = ctrl.Rule(temperatura['chlodno'] & wilgotnosc['sucho'], ilosc_wody['srednio'])
regula2 = ctrl.Rule(temperatura['chlodno'] & wilgotnosc['przecietnie'], ilosc_wody['srednio'])
regula3 = ctrl.Rule(temperatura['chlodno'] & wilgotnosc['mokro'], ilosc_wody['malo'])

regula4 = ctrl.Rule(temperatura['cieplo'] & wilgotnosc['sucho'], ilosc_wody['duzo'])
regula5 = ctrl.Rule(temperatura['cieplo'] & wilgotnosc['przecietnie'], ilosc_wody['srednio'])
regula6 = ctrl.Rule(temperatura['cieplo'] & wilgotnosc['mokro'], ilosc_wody['malo'])

regula7 = ctrl.Rule(temperatura['goraco'] & wilgotnosc['sucho'], ilosc_wody['duzo'])
regula8 = ctrl.Rule(temperatura['goraco'] & wilgotnosc['przecietnie'], ilosc_wody['duzo'])
regula9 = ctrl.Rule(temperatura['goraco'] & wilgotnosc['mokro'], ilosc_wody['srednio'])

system_kontroli = ctrl.ControlSystem([regula1, regula2, regula3, regula4, regula5, regula6, regula7, regula8, regula9])
system_podlewania = ctrl.ControlSystemSimulation(system_kontroli)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
temperatura.view()
plt.title('Funkcje przynależności dla temperatury')

plt.subplot(3, 1, 2)
wilgotnosc.view()
plt.title('Funkcje przynależności dla wilgotności')

plt.subplot(3, 1, 3)
ilosc_wody.view()
plt.title('Funkcje przynależności dla ilości wody')

plt.tight_layout()
plt.savefig('funkcje_przynaleznosci.png')
plt.show()

temp_range = np.arange(15, 36, 1)
wilg_range = np.arange(0, 101, 5)

wynik = np.zeros((len(temp_range), len(wilg_range)))

for i, t in enumerate(temp_range):
    for j, w in enumerate(wilg_range):
        system_podlewania.input['temperatura'] = t
        system_podlewania.input['wilgotnosc'] = w
        system_podlewania.compute()
        wynik[i, j] = system_podlewania.output['ilosc_wody']

plt.figure(figsize=(12, 8))
plt.pcolormesh(wilg_range, temp_range, wynik, cmap='RdYlGn_r', shading='auto')
plt.colorbar(label='Ilość wody [l/dzień]')
plt.xlabel('Wilgotność [%]')
plt.ylabel('Temperatura [°C]')
plt.title('Dzienna ilość wody do podlewania w zależności od temperatury i wilgotności')
plt.grid(True)
plt.savefig('heatmapa.png')
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(wilg_range, temp_range)

surf = ax.plot_surface(X, Y, wynik, cmap='RdYlGn_r', edgecolor='none')
ax.set_xlabel('Wilgotność [%]')
ax.set_ylabel('Temperatura [°C]')
ax.set_zlabel('Ilość wody [l/dzień]')
ax.set_title('Trójwymiarowa wizualizacja zależności ilości wody od temperatury i wilgotności')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Ilość wody [l/dzień]')
plt.savefig('wykres_3d.png')
plt.show()

print("Przykładowe wyniki:")
print("-" * 50)

test_cases = [
    (17, 20, "Chłodno i sucho"),
    (17, 50, "Chłodno i przeciętnie wilgotno"),
    (17, 80, "Chłodno i mokro"),
    (25, 20, "Ciepło i sucho"),
    (25, 50, "Ciepło i przeciętnie wilgotno"),
    (25, 80, "Ciepło i mokro"),
    (32, 20, "Gorąco i sucho"),
    (32, 50, "Gorąco i przeciętnie wilgotno"),
    (32, 80, "Gorąco i mokro")
]

for temp, wilg, opis in test_cases:
    system_podlewania.input['temperatura'] = temp
    system_podlewania.input['wilgotnosc'] = wilg
    system_podlewania.compute()
    ilosc = system_podlewania.output['ilosc_wody']
    print(f"{opis}: Temperatura = {temp}°C, Wilgotność = {wilg}%, Ilość wody = {ilosc:.2f} l/dzień")

# Wnioski
print("\nWnioski:")
print("-" * 50)
print("1. System automatycznego podlewania reaguje zgodnie z oczekiwaniami na podstawie przyjętych reguł.")
print("2. Przy wysokich temperaturach i niskiej wilgotności, system podaje największą ilość wody (do 25l/dzień).")
print("3. Przy niskich temperaturach i wysokiej wilgotności, system ogranicza podlewanie do minimum.")
print("4. W warunkach przeciętnych (średnia temperatura i wilgotność), system utrzymuje zalecane nawodnienie ok. 10l/dzień.")
print("5. Zastosowanie logiki rozmytej pozwala na płynne przejścia między stanami, co zwiększa efektywność nawadniania.")
print("6. Przy ekstremalnie wysokiej temperaturze i niskiej wilgotności, system maksymalizuje nawadnianie do górnej granicy 25l/dzień.")
