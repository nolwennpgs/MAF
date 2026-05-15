%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Correct fisheye
%%%% written by Nolwenn PAGES - LECOB - April 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

import matplotlib
import numpy as np
import cv2
from os import chdir, listdir

repertory = '';
chdir(repertory)

# Charger l'image
image = cv2.imread("XX.JPG")
h, w = image.shape[:2]

# Matrice de l'appareil photo (approximation si pas de calibration réelle)
K = np.array([[w, 0, w / 2],
              [0, w, h / 2],
              [0, 0, 1]])

# Coefficients de distorsion (à ajuster selon l'objectif en faisant des tests)
# k1, k2, p1, p2, k3
# k1: distorsion latérale: k1>0 = agrandissements des côtés; k1 <0 = diminution des côtés
# k2: distorsion haut/bas
# p1: inclinaison haut/bas
# p2, k3
# dist_coeffs = np.array([-0.3, 0.1, 0, 0, 0])
dist_coeffs = np.array([0.7, 0.7, 0, 0, 0])

new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))

# Correction de la distorsion
undistorted = cv2.undistort(image, K, dist_coeffs, None, new_K)

# Rogner l'image pour supprimer les zones externes
x, y, w_roi, h_roi = roi
undistorted = undistorted[y:y + h_roi, x:x + w_roi]

# # Sauvegarde de l'image mise à jour
cv2.imwrite("XX_undistorded.JPG", undistorted)

# Affichage (optionnel)
# cv2.imshow("Original", cv2.resize(image, (487, 557)))

# cv2.imshow("Corrigee", cv2.resize(undistorted, (487, 557)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
