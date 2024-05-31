import cv2
from pyzbar.pyzbar import decode
import numpy as np
import pickle

# Inicializace kamery
cap = cv2.VideoCapture(0)

# Skutečné rozměry QR kódu (v centimetrech)
qr_code_width_cm = 18.7  # Upravte na skutečné rozměry QR kódu

# Rozlišení obrazu kamery
camera_width = 2688
camera_height = 1520

# Načítání kalibračních parametrů
try:
    with open("calibration.pkl", "rb") as f:
        camera_matrix, distortion_coefficients = pickle.load(f)
except FileNotFoundError:
    print("Súbor s kalibráciou nebol nájdený. Skontrolujte, či ste vykonali kalibráciu.")
    exit()

while True:
    ret, frame = cap.read()

    # Zvýšení kontrastu a jasu
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

    # Před zpracováním frame
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Prevod na čiernobiele
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Použití adaptivního prahování
    gray_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    decoded_objects = decode(gray_frame)

    for obj in decoded_objects:
        if obj.type == 'QRCODE':
            points = obj.polygon    # ziska polygon(obrys) qr kodu
            points = np.array(points, dtype=np.int32)   # suradnice rohov sa prevedu na pole pomocou NumPy
            cv2.polylines(frame, [points], True, (0, 255, 0), 2)    # vykresli zeleny obrys 

            # Získejte šířku QR kódu v pixelech
            qr_width_px = abs(points[0][0] - points[1][0])

            # Vypočítejte vzdálenost od kamery k QR kódu s kalibračními parametry
            distance = (qr_code_width_cm * camera_matrix[0, 0]) / qr_width_px

            # Vypočítejte horizontálny a vertikálny uhol vzhľadom na stred obrazu
            horizontal_angle = np.degrees(np.arctan2((points[0][0] + points[1][0]) / 2 - camera_width / 2, camera_matrix[0, 0]))
            vertical_angle = np.degrees(np.arctan2((points[0][1] + points[3][1]) / 2 - camera_height / 2, camera_matrix[1, 1]))

            # Zobrazi vzdialenost vedla obrysu
            cv2.putText(frame, f'Vzdialenost: {2 * distance:.0f} cm', (points[0][0], points[0][1] + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Zobrazi horizontálny uhol vedla obrysu
            cv2.putText(frame, f'Horizontalny uhol: {int(47 + horizontal_angle)} stupnov', (points[0][0], points[0][1] + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Zobrazi vertikálny uhol vedla obrysu
            cv2.putText(frame, f'Vertikalny uhol: {int(50 + vertical_angle)} stupnov', (points[0][0], points[0][1] + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("QR Code Scanner", frame)  # Zde by měl být zobrazen upravený snímek

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Uvolnění zdrojů
cap.release()
cv2.destroyAllWindows()