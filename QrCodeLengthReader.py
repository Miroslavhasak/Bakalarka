import cv2
from pyzbar.pyzbar import decode
import numpy as np
import pickle

cap = cv2.VideoCapture(0)

qr_code_width_cm = 18.7  

camera_width = 2688
camera_height = 1520

try:
    with open("calibration.pkl", "rb") as f:
        camera_matrix, distortion_coefficients = pickle.load(f)
except FileNotFoundError:
    print("Súbor s kalibráciou nebol nájdený. Skontrolujte, či ste vykonali kalibráciu.")
    exit()

while True:
    ret, frame = cap.read()

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    decoded_objects = decode(gray_frame)

    for obj in decoded_objects:
        if obj.type == 'QRCODE':
            points = obj.polygon    
            points = np.array(points, dtype=np.int32)   
            cv2.polylines(frame, [points], True, (0, 255, 0), 2)    

            qr_width_px = abs(points[0][0] - points[1][0])

            distance = (qr_code_width_cm * camera_matrix[0, 0]) / qr_width_px

            horizontal_angle = np.degrees(np.arctan2((points[0][0] + points[1][0]) / 2 - camera_width / 2, camera_matrix[0, 0]))
            vertical_angle = np.degrees(np.arctan2((points[0][1] + points[3][1]) / 2 - camera_height / 2, camera_matrix[1, 1]))

            print(f'Vzdialenost: {2 * distance:.2f} cm')
            print(f'Horizontalny uhol: {47 + horizontal_angle:.2f} stupnov')
            print(f'Vertikalny uhol: {50 + vertical_angle:.2f} stupnov')

            cv2.putText(frame, f'Vzdialenost: {2 * distance:.0f} cm', (points[0][0], points[0][1] + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, f'Horizontalny uhol: {47 + horizontal_angle:.2f} stupnov', (points[0][0], points[0][1] + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, f'Vertikalny uhol: {50 + vertical_angle:.2f} stupnov', (points[0][0], points[0][1] + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("QR Code Scanner", frame)  

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()
