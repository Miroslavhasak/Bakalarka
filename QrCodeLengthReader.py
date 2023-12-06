import cv2
from pyzbar.pyzbar import decode
import numpy as np
import pickle

# Inicializace kamery
cap = cv2.VideoCapture(0)

# Proměnná na sledování, zda má začít skenovat QR kódy
start_scanning = False

# Skutečné rozměry QR kódu (v centimetrech)
qr_code_width_cm = 18.7  # Upravte na skutečné rozměry QR kódu

# Rozlišení obrazu kamery
camera_width = 2688
camera_height = 1520

# Načítanie kalibračných parametrov
try:
    with open("calibration.pkl", "rb") as f:
        camera_matrix, distortion_coefficients = pickle.load(f)
except FileNotFoundError:
    print("Súbor s kalibráciou nebol nájdený. Skontrolujte, či ste vykonali kalibráciu.")
    exit()

while True:
    ret, frame = cap.read()

    # Prevod na čiernobiele
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    decoded_objects = decode(gray_frame)

    if start_scanning:
        for obj in decoded_objects:
            # print('Typ: ', obj.type)
            print('Obsah: ', obj.data.decode('utf-8'))

            if obj.type == 'QRCODE':
                points = obj.polygon
                points = np.array(points, dtype=np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 0), 2)

                # Získejte šířku QR kódu v pixelech
                qr_width_px = abs(points[0][0] - points[1][0])

                # Vypočítejte vzdálenost od kamery k QR kódu s kalibračními parametry
                undistorted_points = cv2.undistortPoints(np.array([[[points[0][0], points[0][1]]]], dtype=np.float32), camera_matrix, distortion_coefficients)
                distance = (qr_code_width_cm * camera_matrix[0, 0]) / qr_width_px

                print(f'Vzdálenost od kamery k QR kódu: {distance:.2f} cm')

                # Po nalezení prvního QR kódu ukončit skenování
                start_scanning = False

    cv2.imshow("QR Code Scanner", gray_frame)  # Zde by měl být zobrazen černobílý snímek

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == ord('s'):
        start_scanning = True

# Uvolnění zdrojů
cap.release()
cv2.destroyAllWindows()
