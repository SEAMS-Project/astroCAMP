import subprocess
import keyboard
from time import sleep


def measure():
    commande = [
        "cat",
        "/sys/devices/platform/ams/hwmon/hwmon0/subsystem/hwmon2/power1_input",
    ]
    print("Measuring power under load")
    print(" ctrl-c to end measurement")
    nb = 0
    mesures = 0
    try:
        while True:
            resultat = subprocess.run(commande, capture_output=True, text=True)
            mesures += int(resultat.stdout)
            nb += 1
            sleep(1)
    except KeyboardInterrupt:
        print("\n")
        print("Measurement done")
    return mesures / nb


def reference():
    commande = [
        "cat",
        "/sys/class/power_supply/BAT0/power_now",
    ]
    print("Measuring base comsumption")
    nb = 0
    mesures = 0
    for i in range(30):
        resultat = subprocess.run(commande, capture_output=True, text=True)
        mesures += int(resultat.stdout)
        nb += 1
        sleep(1)
    print("Load-less measurement done")
    return mesures / nb


reference = reference()
print("Base power : ", int(reference), "µW")

moyenne = measure()
print("average : ", int(moyenne), "µW")
print("Average power overhead under load : ", int((moyenne - reference) / 1000), "mW")
