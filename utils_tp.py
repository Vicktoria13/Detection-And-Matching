import numpy as np
import PIL
import cv2 as cv

import matplotlib.pyplot as plt
import  random
random.seed(42)




######## CONSTANTES MASQUES ########

masqueX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
masqueY =np.array([[-1,-2,-1], [0, 0, 0], [1, 2, 1]])
masque_gaussien_formula = np.zeros((3,3))
sigma = 1
mu = 3
pas = 1
for i in range(3):
    for j in range(3):
        masque_gaussien_formula[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-((i-mu)**2+(j-mu)**2)/(2*sigma**2))
        
masque_gaussien_formula = masque_gaussien_formula/np.sum(masque_gaussien_formula)


#########################################





def gris_moyen(image):
    """
    Convertit une image en niveau de gris 

    Cv::Mat --> Cv::Mat

    """
    cop = image.copy()
    gris = cv.cvtColor(cop, cv.COLOR_BGR2GRAY)
    return gris


def affiche_image(name, src):

    """
    Affiche une image 

    Cv::Mat 
    
    """


    cv.imshow(name, src)
    cv.waitKey(0)
    cv.destroyAllWindows()





def print_corners(corner_list, image_source, color = (0,0,255)):

    """
    Affiche les coins detectés sur l'image source dans la couleur souhaitée
    
    """
    
    for indices in corner_list :
        #cv.circle(image_source, (indices[0], indices[1]), 5, (0,0,255), 2)
        #image_source[indices[1], indices[0]] = color
        #tracer des croix
        cv.line(image_source, (indices[0]-5, indices[1]), (indices[0]+5, indices[1]), color, 1)
        cv.line(image_source, (indices[0], indices[1]-5), (indices[0], indices[1]+5), color, 1)
        #þracer cercles
        #cv.circle(image_source, (indices[0], indices[1]), 3, color, 2)

    return image_source
    
    


    

def rotate_image(src,angle):

    """
    Fait une rotation de l'image source d'un angle donné
    """

    cop = src.copy()

    center = (cop.shape[0]//2,cop.shape[1]//2)
    matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    image_apres_rotation = cv.warpAffine(cop, matrix, (cop.shape[1],cop.shape[0]))


    return image_apres_rotation




#l'image doit etre en niveau de gris
def gradients(img):
    """
    Renvoie les gradients de l'image source utilisés pour le detecteur de Harris
    """
    Ix, Iy = np.gradient(img)
    Ix_carre = np.square(Ix)
    Iy_carre = np.square(Iy)
    Ixy = np.multiply(Ix, Iy)
    return Ix, Iy, Ix_carre, Iy_carre, Ixy




def suppression_non_maxima_avant_seuillage(C_harris,taille_fenetre):

    """
    Supprime avant seuillage les non maxima locaux dans la matrice des C
    """
    #size_x, size_y = C_harris.shape[0], C_harris.shape[1]

    #for ligne in range(0, size_x - taille_fenetre + 1,1):
    #    for colonne in range(0, size_y - taille_fenetre + 1,1):

    #        window = C_harris[ligne:ligne+taille_fenetre, colonne:colonne+taille_fenetre]
    #        max = window.max()
    #        if max != C_harris[ligne, colonne]:
    #           C_harris[ligne, colonne] = 0

    C_orig = np.zeros(C_harris.shape)
    size_x, size_y = C_harris.shape[0], C_harris.shape[1]

    #on ne s'interesse que ceux loin des bords 

    start_x_point = taille_fenetre//2
    end_x_point = size_x - taille_fenetre//2
    start_y_point = taille_fenetre//2
    end_y_point = size_y - taille_fenetre//2

    for ligne in range(start_x_point, end_x_point,1):
        for colonne in range (start_y_point, end_y_point,1):
            window = C_harris[ligne:ligne+taille_fenetre, colonne:colonne+taille_fenetre]
            max = window.max()
            if max != C_harris[ligne, colonne]:
                C_orig[ligne, colonne] = 0
            else:
                C_orig[ligne, colonne] = C_harris[ligne, colonne]


    return C_orig

            




def Harris_rectangle(img_src, Ix_carre, Iy_carre, Ixy, seuil, k, flag_suppression_non_maxima, taille_fenetre, offset,couleur):
    

    """
    Retourne l'image source avec les coins detectés et la liste des coins detectés
    img_src : image source RGB
    Ix_carre, Iy_carre, Ixy : gradients de l'image source
    seuil : % du max de C_harris pour le seuillage
    k : parametre de Harris
    flag_suppression_non_maxima : si True, on supprime les non maxima locaux
    taille_fenetre : taille de la fenetre pour le calcul de C_harris
    offset : pas de la fenetre
    couleur : couleur des coins detectés

    """
    hauteur, largeur = img_src.shape[0], img_src.shape[1]
    #hauteur  = nombre de lignes
 
    img_gray = gris_moyen(img_src)
    C_harris = np.zeros(img_gray.shape) #matrice des C

    corner_list = []
    for y in range(0,hauteur - taille_fenetre + 1, offset):
        for x in range(0,largeur - taille_fenetre + 1, offset):
                
                #on prend la fenetre
                windowIxx = Ix_carre[y:y+taille_fenetre, x:x+taille_fenetre]
                windowIxy = Ixy[y:y+taille_fenetre, x:x+taille_fenetre]
                windowIyy = Iy_carre[y:y+taille_fenetre, x:x+taille_fenetre]
    
                Somme_xx = windowIxx.sum()
                Somme_xy = windowIxy.sum()
                Somme_yy = windowIyy.sum()
    
                det = (Somme_xx * Somme_yy) - (Somme_xy**2)
                trace = Somme_xx + Somme_yy
    
                C = det - k * (trace**2)
                C_harris[y, x] = C
    
    if flag_suppression_non_maxima:
        winsize= 20
        C_harris = suppression_non_maxima_avant_seuillage(C_harris,winsize)

    seuil = C_harris.max() * seuil #0.015 #+ tu montes -> - de coins
    C_harris = C_harris > seuil # tableau 2D de booléens

    indices = np.where(C_harris)

    indices_lignes, indices_colonnes = indices
    for i in range(len(indices_lignes)):
        ligne = indices_lignes[i]
        colonne = indices_colonnes[i]
        corner_list.append([colonne, ligne])
    
    return print_corners(corner_list, img_src.copy(),couleur), corner_list
    




def Harris_gaussienne(src, Ix_carre, Iy_carre, Ixy, masque_gaussien,k,flag_suppression_non_maxima,coeff_seuil,couleur):


    """

    Retourne l'image source avec les coins detectés et la liste des coins detectés

    """

    liste_coins = []
    grey = gris_moyen(src)

    #on les convolue avec un masque gaussien
    Ix_carre_gaussienn = cv.filter2D(Ix_carre, -1, masque_gaussien)
    Iy_carre_gaussien = cv.filter2D(Iy_carre, -1, masque_gaussien)
    Ixy_gaussien = cv.filter2D(Ixy, -1, masque_gaussien)

    C_harris = np.multiply(Ix_carre_gaussienn, Iy_carre_gaussien) - np.square(Ixy_gaussien) - k * np.square(Ix_carre_gaussienn + Iy_carre_gaussien)

    if flag_suppression_non_maxima:
        winsize= 20
        C_harris = suppression_non_maxima_avant_seuillage(C_harris,winsize)


    thre = C_harris.max() * coeff_seuil
    C_harris = C_harris > thre # tableau 2D de boolée
    indices = np.where(C_harris)
    indices_lignes, indices_colonnes = indices

    for i in range(len(indices_lignes)):
        ligne = indices_lignes[i]
        colonne = indices_colonnes[i]
        liste_coins.append([colonne, ligne])
    

    return print_corners(liste_coins, src.copy(),couleur), liste_coins
    



#####################################  fast detector

def circle_around_pixel(img, i, j, seuil):

    """
    Renvoie une liste des 16 pixels autour du pixel (i,j) dans un rayon de 3 pixels
    de manière circulaire. La liste est binarisée selon un seuil donné
    """
   
    list_pixels = []
    
    list_pixels.append(img[i-3][j])
    list_pixels.append(img[i-3][j+1])
    list_pixels.append(img[i-2][j+2])
    list_pixels.append(img[i-1][j+3
                                ])
    list_pixels.append(img[i][j+3])
    list_pixels.append(img[i+1][j+3])
    list_pixels.append(img[i+2][j+2])
    list_pixels.append(img[i+3][j+1])

    list_pixels.append(img[i+3][j])
    list_pixels.append(img[i+3][j-1])
    list_pixels.append(img[i+2][j-2])
    list_pixels.append(img[i+1][j-3])

    list_pixels.append(img[i][j-3])
    list_pixels.append(img[i-1][j-3])
    list_pixels.append(img[i-2][j-2])
    list_pixels.append(img[i-3][j-1])


    ref = img[i][j]

    for k in range(len(list_pixels)):
        if list_pixels[k]<ref-seuil or list_pixels[k]>ref+seuil:
            list_pixels[k]=1
        else:
            list_pixels[k]=0
    

    return list_pixels




def nb_un_successifs(V):
    score_depart = 0   # Sauvegarde du score du début du vecteur
    init_score_depart = False

    score = 0           # Score courant
    meilleur = 0        # Meilleur score

    for e in V :
        if (e == 1) : score += 1        # On ajoute 1 à chaque 1 successif
        else :                          # Sinon, on vérifie le record et on retombe à 0
            # Vérification du record
            if (score > meilleur) : meilleur = score
            
            # Sauvegarde du premier score
            if (init_score_depart == False) :
                score_depart = score
                init_score_depart = True
        
            # Remise à zéro
            score = 0

    # Vérification si le score en fin de vecteur et celui en début sont au-dessus du record
    if ((score+score_depart) > meilleur) : meilleur = (score+score_depart)

    return meilleur

def trouver_n_1_successifs(vecteur, n):
    """
    Renvoie True si le vecteur contient n 1 successifs, False sinon via la convolution
    """
    vecteur_convolution = np.ones(n, dtype=int)
    
    resultat_convolution = np.convolve(vecteur, vecteur_convolution, mode='valid')
    indices = np.where(resultat_convolution >= n)[0]
    
    if len(indices) > 0:
        return True 
    else:
        return False  
    





def Fast_detector(src, seuil_voisins,seuil_fast):

    """
    Retourne l'image source avec les coins detectés et la liste des coins detectés
    """

    #'img' est une image en niveau de gris
    image_apres_Fast = src.copy()
    img_grey = gris_moyen(image_apres_Fast)
    liste_corners_Fast = []

    #on ne s'interesse pas aux bords
    for i in range(3, src.shape[0]-3):
        for j in range(3, src.shape[1]-3):
            list_pixels = circle_around_pixel(img_grey, i, j, seuil_fast)
            flag = trouver_n_1_successifs(list_pixels, seuil_voisins)
            if flag:

                liste_corners_Fast.append([j, i])
                image_apres_Fast[i][j]=[0,0,255]

        
    return print_corners(liste_corners_Fast, image_apres_Fast.copy(),(0,0,255)), liste_corners_Fast



def Fast_detector_avec_score(src, seuil_nb_voisins,seuil_FAST):


    """
    Retourne l'image source avec les coins detectés et la liste des coins detectés
    Avec la suppression des non maxima locaux implémentée
    """


    #'img' est une image en niveau de gris
    image_apres_Fast = src.copy()
    img_grey = gris_moyen(src)
    liste_corners_Fast = []

    #liste score : chaque pixel aura un score a associé au nombre de 1 successifs
    score = np.zeros(img_grey.shape)

    for i in range(3, src.shape[0]-3):
        for j in range(3, src.shape[1]-3):
            list_pixels = circle_around_pixel(img_grey, i, j, seuil_FAST)
            nb_1_successifs = nb_un_successifs(list_pixels)

        
            #si c'est un coin detecté
            if nb_1_successifs >=seuil_nb_voisins:
                score[i][j] = nb_1_successifs
            else:
                score[i][j] = 0


    #on supprime les non maxima locaux avec score ==> nb de 1 successifs


    winsize = 20
    score = suppression_non_maxima_avant_seuillage(score, winsize)

    #seuil
    score = score > 10
    indices = np.where(score)
    indices_lignes, indices_colonnes = indices

    for i in range(len(indices_lignes)):
        ligne = indices_lignes[i]
        colonne = indices_colonnes[i]
        liste_corners_Fast.append([colonne, ligne])
        image_apres_Fast[ligne][colonne]=[0,0,255]

    
    return print_corners(liste_corners_Fast, image_apres_Fast.copy(),(0,0,255)), liste_corners_Fast




############################################# Description



def vecteur_autour_point_interet(src, liste_corners, pos_liste, taille_bloc):

    #src en niveau de gris normalisé entre 0 et 1
    """
    Renvoie le vecteur de taille taille_bloc*taille_bloc autour du point d'interet
    """
    vecteur = []

    key_point = liste_corners[pos_liste]
    xp, yp = key_point[0], key_point[1]

    #on prend la fenetre
    window = src[yp-taille_bloc//2:yp+taille_bloc//2, xp-taille_bloc//2:xp+taille_bloc//2]
    window = window.flatten()
    return window



#### LBP

def minimal_lbp_code(binary_code):
    num_bits = len(binary_code)
    min_code = binary_code.copy()
    for shift in range(1, num_bits):
        rotated_code = binary_code[-shift:] + binary_code[:-shift]
        if rotated_code < min_code:
            min_code = rotated_code

    #reshape en 16*1
    min_code = np.array(min_code)
    min_code = min_code.flatten()
    return min_code



def extract_neighborhood(image, position):
    x, y = position
    if x < 1 or y < 1 or x > image.shape[0] - 2 or y > image.shape[1] - 2:
        raise ValueError("La position est trop proche du bord de l'image.")

    neighborhood = []
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            neighborhood.append(image[i, j])

    return np.array(neighborhood)




def vecteur_autour_point_interet_LBP(img, liste_corners, pos_liste):

    #src en niveau de gris (normalisé entre 0 et 1 si possible)
    """
    Renvoie le vecteur de taille taille_bloc*taille_bloc autour du point d'interet
    """
    vecteur = []

    key_point = liste_corners[pos_liste]


    #on prend la fenetre
    list_pixels = extract_neighborhood(img, (key_point[1], key_point[0]))

    #pixel principal
    I_PO = img[key_point[1], key_point[0]]

    for k in range(len(list_pixels)):
        if list_pixels[k]<I_PO:
            vecteur.append(0)
        else:
            vecteur.append(1)

    #pour ajouter l'invariance en rotation
    vecteur = minimal_lbp_code(vecteur)

    return vecteur




def hamming_distance(descriptor1, descriptor2):
    if len(descriptor1) != len(descriptor2):
        raise ValueError("Les descripteurs doivent avoir la même longueur")

    distance = sum(bit1 != bit2 for bit1, bit2 in zip(descriptor1, descriptor2))
    return distance




def l1_distance(descriptor1, descriptor2):
    """
    Renvoie la distance L1 entre deux descripteurs
    """
    # Assurez-vous que les deux descripteurs ont la même taille

    if( len(descriptor1) != len(descriptor2)) :
        raise ValueError("Les descripteurs doivent avoir la même longueur")
    
    # Calcul de la distance L1
    dif = abs(descriptor1 - descriptor2)
    distance = np.sum(dif)

    return distance


def l2_distance(descriptor1, descriptor2):
    """
    Renvoie la distance L2 entre deux descripteurs
    """


    if( len(descriptor1) != len(descriptor2)) :
        raise ValueError("Les descripteurs doivent avoir la même longueur")
    
    # Calcul de la distance L2
    dif = abs(descriptor1 - descriptor2)
    distance = np.sqrt(np.sum(dif**2))

    return distance



def correlation_classique(descriptor1, descriptor2):
    """
    Renvoie la distance L2 entre deux descripteurs
    """

    if len(descriptor1) != len(descriptor2):
        raise ValueError("Les descripteurs doivent avoir la même longueur")

    descriptor1 = np.array(descriptor1)
    descriptor2 = np.array(descriptor2)

    num = np.dot(descriptor1, descriptor2)
    den = np.linalg.norm(descriptor1) * np.linalg.norm(descriptor2)

    corr = num / den
    #si corr = 1, alors les deux descripteurs sont identiques
    #si corr = 0, alors les deux descripteurs sont totalement différents

    #rend entre 0 et 1
    return 1 - corr


# pour eviter le changement d'intensité lumineuse
# a comparer avec distance L1
#si 0, alors les deux descripteurs sont identiques
#si 1, alors les deux descripteurs sont totalement différents

def correlation_croisee_normalisee(descripteur1, descripteur2):

    if len(descripteur1) != len(descripteur2):
        raise ValueError("Les descripteurs doivent avoir la même longueur")

    descripteur1 = np.array(descripteur1)
    descripteur2 = np.array(descripteur2)

    # Calcul de la moyenne des deux descripteurs
    moyenne1 = np.mean(descripteur1)
    moyenne2 = np.mean(descripteur2)

    # Calcul de la corrélation croisée
    correlation_croisee = np.sum((descripteur1 - moyenne1) * (descripteur2 - moyenne2))

    # Normalisation de la corrélation croisée
    denominateur = np.sqrt(np.sum((descripteur1 - moyenne1) ** 2) * np.sum((descripteur2 - moyenne2) ** 2))
    
    if denominateur == 0:
        return 1000000000 # Évitez la division par zéro et de le considérer comme un mauvais match : on l'enleve
    
    correlation_croisee_normalisee = correlation_croisee / denominateur

    
    return 1-correlation_croisee_normalisee




def metrique_entre_deux_descripteurs(descriptor1, descriptor2, operation_metrique):
    
    res = operation_metrique(descriptor1, descriptor2)
    return res


def matching(image1, image2, liste_coins1, liste_coins2, taille_fenetre, operation_metrique,ratio):
    
    """
    Renvoie la liste des matchs entre deux images
    """
    #doivent etre en niveaux de gris  SI POSSIBLE NORMALISES ENTRE 0 ET 1
    
    #def filtre_coins_sur_bords(img, list_corners_detected, taille_bloc):

    l1 = filtre_coins_sur_bords(image1, liste_coins1, taille_fenetre)
    l2 = filtre_coins_sur_bords(image2, liste_coins2, taille_fenetre)
  
    liste_match = [] #est supposé contenir les matchs entre les deux images
    #par exemple :
    #liste_match = [[pos_coins_match1, pos_coins_match2, distance], [pos_coins_match1, pos_coins_match2, distance], ...]
    #signifie que le point d'indice pos_coins_match1 dans la liste des coins de l'image 1 correspond 
    # au point d'indice pos_coins_match2 dans la liste des coins de l'image 2


    distance_min_1 = 100000000
    distance_min_2 = 100000000

    i1 = image1.copy()
    i2 = image2.copy()



    indice1 = 0


    for j in range(len(l1)):

        #on calcule son descripteur : rend un vecteur de taille taille_fenetre*taille_fenetre elements
        descripteur1 = vecteur_autour_point_interet(image1, l1, j, taille_fenetre)

        for k in range(len(l2)):

            descripteur2 = vecteur_autour_point_interet(image2, l2, k, taille_fenetre)
            d = metrique_entre_deux_descripteurs(descripteur1, descripteur2, operation_metrique)

            #on regarde si la distance consideré est petite a nos 2 distances minimales
            if d < distance_min_1 or d < distance_min_2:

                if d < distance_min_1:
                    distance_min_1 = d
                    indice1 = k #on garde que l'indice du point le + proche
                else:
                    distance_min_2 = d


        if distance_min_1/distance_min_2 < ratio :

         
            match_found = [l1[j][0], l1[j][1], l2[indice1][0], l2[indice1][1]]
            liste_match.append(match_found)

            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            cv.circle(i1, (l1[j][0], l1[j][1]), 5, color, 2)
            cv.circle(i2, (l2[indice1][0], l2[indice1][1]), 5, color, 2)

         

    
        distance_min_1 = 900000000
        distance_min_2 = 900000000


    print("len liste_match = ", len(liste_match))
    return liste_match, i1, i2





def matching_invariant(image1, image2, liste_coins1, liste_coins2, taille_fenetre, operation_metrique,ratio):
    
    """
    Renvoie la liste des matchs entre deux images
    """
    #doivent etre en niveaux de gris  SI POSSIBLE NORMALISES ENTRE 0 ET 1
    
    #def filtre_coins_sur_bords(img, list_corners_detected, taille_bloc):

    l1 = filtre_coins_sur_bords(image1, liste_coins1, taille_fenetre)
    l2 = filtre_coins_sur_bords(image2, liste_coins2, taille_fenetre)
  
    liste_match = [] #est supposé contenir les matchs entre les deux images

    distance_min_1 = 100000000
    distance_min_2 = 100000000

    i1 = image1.copy()
    i2 = image2.copy()



    indice1 = 0


    for j in range(len(l1)):

        #on calcule son descripteur : rend un vecteur de taille taille_fenetre*taille_fenetre elements
        descripteur1 = vecteur_autour_point_interet_LBP(image1, l1, j)

        for k in range(len(l2)):

            descripteur2 = vecteur_autour_point_interet_LBP(image2, l2, k)
            d = abs(descripteur1 - descripteur2).sum()


            #on regarde si la distance consideré est petite a nos 2 distances minimales
            if d < distance_min_1 or d < distance_min_2:

                if d < distance_min_1:
                    distance_min_1 = d
                    indice1 = k #on garde que l'indice du point le + proche
                else:
                    distance_min_2 = d


        if distance_min_1/distance_min_2 < ratio :

            print("distance_min_1", distance_min_1)
            match_found = [l1[j][0], l1[j][1], l2[indice1][0], l2[indice1][1]]
            liste_match.append(match_found)

            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            cv.circle(i1, (l1[j][0], l1[j][1]), 5, color, 2)
            cv.circle(i2, (l2[indice1][0], l2[indice1][1]), 5, color, 2)

         

           

        distance_min_1 = 900000000
        distance_min_2 = 900000000


    print("len liste_match = ", len(liste_match))
    return liste_match, i1, i2
    



def niveaux_de_gris_entre_0_et_1(img):
    """
    Normalise les valeurs de l'image entre 0 et 1 pour gagnger en temps de calcul
    """
    img = img.astype(float)
    img = img / 255

    return img




def filtre_coins_sur_bords(img, list_corners_detected, taille_bloc):
    """
    Renvoie la liste des coins detectés en enlevant les coins trop proches des bords
    """
    #on enleve les coins qui sont trop proches des bords
    list_corners = list_corners_detected.copy()
    for indices in list_corners_detected :
        x = indices[0]
        y = indices[1]

        if x < taille_bloc//2 or y < taille_bloc//2 or x > img.shape[1] - taille_bloc//2 or y > img.shape[0] - taille_bloc//2:
            list_corners.remove(indices)


    return list_corners



# sert a afficher les matchs entre deux images
def print_matching(img1, img2, liste_coins1, liste_coins2, liste_match, pas,titre_graphe):

    """
    Affiche les matchs entre deux images via des cercles de couleurs differentes
    """

    src1 = img1.copy()
    src2 = img2.copy()
    random.seed(42)

    for i in range(0,len(liste_match),pas):
        #on recupere les points de match
        liste_couleur = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

        point1 = liste_coins1[liste_match[i][0]]
        point2 = liste_coins2[liste_match[i][1]]


        #tracer croix
        cv.line(src1, (point1[0]-5, point1[1]), (point1[0]+5, point1[1]), liste_couleur, 2)
        cv.line(src1, (point1[0], point1[1]-5), (point1[0], point1[1]+5), liste_couleur, 2)

        cv.line(src2, (point2[0]-5, point2[1]), (point2[0]+5, point2[1]), liste_couleur, 2)
        cv.line(src2, (point2[0], point2[1]-5), (point2[0], point2[1]+5), liste_couleur, 2)
        
    
    #on affiche les deux images sur matplotlib

    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.imshow(src1)
    plt.title("image 1")
    plt.subplot(1,2,2)
    plt.imshow(src2)
    plt.title("image 2")
    plt.suptitle(titre_graphe)
    plt.tight_layout()
    plt.show()



def print_corners_with_concat_lines(src1, src2, list_match, pas, titre_graphe):

    img1 = src1.copy()
    img2 = src2.copy()

    #concatenation des deux images
    img_concat = np.concatenate((img1, img2), axis=1)
    img_concat = cv.cvtColor(img_concat, cv.COLOR_BGR2RGB)

    #on trace les lignes entre les points de match
    #sachant que liste_match = [[x1_M1,y1_M1,x1_M2,y1_M2], [x2_M1,y2_M1,x2_M2,y2_M2], ...]

    for i in range(0,len(list_match),pas):
        #on recupere les points de match
        liste_couleur = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

        x1 = list_match[i][0]
        y1 = list_match[i][1]

        x2 = list_match[i][2]
        y2 = list_match[i][3]

        point1 = [x1, y1]
        point2 = [x2 + img1.shape[1], y2]

        #tracer ligne
        cv.line(img_concat, (point1[0], point1[1]), (point2[0], point2[1]), liste_couleur, 2)

    plt.figure(figsize=(15,15))
    plt.imshow(img_concat)
    plt.title(titre_graphe)
    plt.show()

############################### test

def test_influence_taille_fenetre(src_image, Ix_carre, Iy_carre, Ixy, seuil, k,taille_fenetre,couleur):
    #res, list_corners_no_supp_rect = Harris_rectangle(src_image, Ix_carre, Iy_carre, Ixy,seuil,k,False,taille_fenetre ,deplacement_fenetre,couleur)

    
    """#influence et evolution de la taille de la fenetre
    """
    #fixe
    deplacement_fenetre = 2
    n = len(taille_fenetre)

    plt.figure(figsize=(15,15))
    for i in range(len(taille_fenetre)):
        res, liste_cor = Harris_rectangle(src_image.copy(), Ix_carre, Iy_carre, Ixy,seuil,k,True,taille_fenetre[i] ,deplacement_fenetre,couleur)
        res = cv.cvtColor(res, cv.COLOR_BGR2RGB)

        
        plt.subplot(1,n,i+1)
        plt.imshow(res)
        plt.title("Harris+rectangle SIZE="+str(taille_fenetre[i])+ " et nb_points = "+str(len(liste_cor)), size=10)

    plt.show()




def test_influence_k(src_image, Ix_carre, Iy_carre, Ixy, seuil, k,couleur):
    """
    influence et evolution de la taille de la fenetre
    """
    n = len(k)

    plt.figure(figsize=(15,15))
    for i in range(len(k)):
        res, list_corners = Harris_gaussienne(src_image.copy(), Ix_carre, Iy_carre, Ixy, masque_gaussien_formula,k[i],True,seuil,couleur)
        res = cv.cvtColor(res, cv.COLOR_BGR2RGB)

        
        plt.subplot(1,n,i+1)
        plt.imshow(res)
        plt.title("Harris Gaussienne avec k ="+str(k[i])+" et nb_points = "+str(len(list_corners)), size=10)

    plt.show()





def test_res_Harris_OPENCV(src, your_result):
    """
    #Test avec fonction déja implémentés
    """
    gray = gris_moyen(src)
    corners = cv.cornerHarris(gray,2,3,0.04)

    src[corners>0.01*corners.max()]=[0,0,255]

    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    #comparaison entre mon implémentation et celle de opencv
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(your_result)
    plt.title("Mon implémentation du detecteur de Harris ")
    plt.subplot(1,2,2)
    plt.imshow(src)
    plt.title("detecteur de Harris d'opencv")


def test_res_FAST_OPENCV(src,your_result):
    """
    #Test avec fonction déja implémentés
    """

    img = src.copy()
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    # find and draw the keypoints
    kp = fast.detect(img,None)
    img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255))

    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    your_result = cv.cvtColor(your_result, cv.COLOR_BGR2RGB)
    #comparaison entre mon implémentation et celle de opencv
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(your_result)
    plt.title("Mon implémentation du detecteur Fast ")
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.title("detecteur Fast d'opencv")