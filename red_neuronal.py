import numpy as np
from PIL import Image # Funcion para Imagenes.
import scipy.special
import matplotlib.pyplot as plt
import time

# Funcion para convertir una Imagen.
def convert(nombre):
    f = Image.open(nombre) # Abre la imagen.
    rgb_im = f.convert('RGB') # convierte a rgb.
    width, height = rgb_im.size # obtiene dimensiones.
    mat = np.zeros((width, height)) # arreglo con mismas dimensiones.
    im = rgb_im
    pix = im.load()
    grises = [] # Lista vacia que va a contener al promedio de las componente RGB.
    for x in range(0,width): # Para cada pixel obtengo su promedio de las 3 componentes RGB.
        for y in range(0,height):
            r, g, b = rgb_im.getpixel((x,y))
            gris = float(float(r+g+b)/3.0)/255.
            grises.append(gris)
            mat[x,y] = gris
    tam = len(grises)
    patron = np.array(np.zeros((tam,1),float))
    for i in range(0,tam):
        patron[i,0] = float(grises[i])
    
    return patron # Devuelve un vector de una columa con los avlores de la imagen

# Valor entrenamiento (valores de entrada).
inicio = time.time() 

# Nuestro alpha.
a = 0.4

# Imagenes de Entrenamiento de Ciudades.
En = np.array(np.mat(convert("Imagenes Entrenamiento/Ciudad 1.jpg")))
En_2 = np.array(np.mat(convert("Imagenes Entrenamiento/Ciudad 2.jpg")))
En_3 = np.array(np.mat(convert("Imagenes Entrenamiento/Ciudad 3.jpg")))
En_4 = np.array(np.mat(convert("Imagenes Entrenamiento/Ciudad 4.jpg")))

# Imagenes de Entrenamiento de Paisajes.
En2 = np.array(np.mat(convert("Imagenes Entrenamiento/Paisaje 1.jpg")))
En2_2 = np.array(np.mat(convert("Imagenes Entrenamiento/Paisaje 2.jpg")))
En2_3 = np.array(np.mat(convert("Imagenes Entrenamiento/Paisaje 3.jpg")))
En2_4 = np.array(np.mat(convert("Imagenes Entrenamiento/Paisaje 4.jpg")))

# Imagenes de Prueba.
img1_pr = np.array(np.mat(convert("Imagenes Entrenamiento/Prueba 1.jpg")))
img2_pr = np.array(np.mat(convert("Imagenes Entrenamiento/Prueba 2.jpg")))
img3_pr = np.array(np.mat(convert("Imagenes Entrenamiento/Prueba 3.jpg")))
img4_pr = np.array(np.mat(convert("Imagenes Entrenamiento/Prueba 5.jpg")))

# Funcion para concatenar el Bias.
def ConBias(x):
	bias = -np.ones((1, 1))
	x = np.concatenate([x, bias], axis = 0)
	return x

# Imagenes de Entrenamiento de Ciudades cargando Bias.
En = ConBias(En)
En_2 = ConBias(En_2)
En_3 = ConBias(En_3)
En_4 = ConBias(En_4)

# Imagenes de Entrenamiento de Paisajes cargando Bias.
En2 = ConBias(En2)
En2_2 = ConBias(En2_2)
En2_3 = ConBias(En2_3)
En2_4 = ConBias(En2_4)

# Imagenes de Prueba cargando Bias.
img1_pr = ConBias(img1_pr)
img2_pr = ConBias(img2_pr)
img3_pr = ConBias(img3_pr)
img4_pr = ConBias(img4_pr)

# Muestra de los Datos de Ciudades.
print("\tMATRICES PARA CIUDADES")
print("Matriz 1 de entrenamiento:")
print(En)
print("Matriz 2 de entrenamiento:")
print(En_2)
print("Matriz 1 de entrenamiento:")
print(En_3)
print("Matriz 2 de entrenamiento:")
print(En2_4)
# Muestra de los Datos de Paisajes.
print("\n\tMATRICES PARA PAISAJES")
print("Matriz 1 de entrenamiento:")
print(En)
print("Matriz 2 de entrenamiento:")
print(En_2)
print("Matriz 1 de entrenamiento:")
print(En_3)
print("Matriz 2 de entrenamiento:")
print(En2_4)

# Valores objetivos.
Ob = np.zeros((2,1))
Ob[0] = 1
Ob2 = np.zeros((2,1))
Ob2[1] = 1

# Nuestros pesos.
w1 = np.random.rand(100, len(En))
w2 = np.random.rand(2, 100)

# Funcion sigmoide.
def sigmoid(x):
    return scipy.special.expit(x) # Genera la funcion sigmoide

# Valores de Lambdas.
lamb1 = 0.05
lamb2 = 0.01

# Funcion de entrenamiento.
def entrena(w1, w2, En, Ob):
    # Capa intermedia.
    Si = sigmoid(np.dot(w1, En))
    # Capa de Salida.
    S = sigmoid(np.dot(w2, Si))
    # Optenemos el error.
    e = Ob-S
    ei = np.dot(w2.T, e)
    # Actualizamos los pesos.
    w2 += a* (np.dot(e*S*(1.-S), Si.T) - lamb2 * w2) # Actualizacion de los pesos.
    w1 += a* (np.dot(ei*Si*(1.-Si), En.T) - lamb1 * w1) # Actualizacion de los pesos.
    return w1, w2, np.mean(e) # Promedio de los 3 errores.

# Esta funcion nos clasifica los patrones con los valotes de entrada.
def Clasifica(w1, w2, En):
    Si = sigmoid(np.dot(w1, En))
    S = sigmoid(np.dot(w2, Si))
    return S

print("\nENTRENANDO ....\n")

list_error = []
# Hacemos que aprendan los dos conjuntos de salidos y con nuestros pesos.
for i in range(0, 100):
    w1, w2, error = entrena(w1, w2, En, Ob)
    w1, w2, error = entrena(w1, w2, En2, Ob2)
    w1, w2, error = entrena(w1, w2, En_2, Ob)
    w1, w2, error = entrena(w1, w2, En2_2, Ob2)
    w1, w2, error = entrena(w1, w2, En_3, Ob)
    w1, w2, error = entrena(w1, w2, En2_3, Ob2)
    w1, w2, error = entrena(w1, w2, En_4, Ob)
    w1, w2, error = entrena(w1, w2, En2_4, Ob2)
    list_error.append(abs(error)) # Valores absolutos
# Muestra de los Objetivos.

# Mostramos los valores para Ciudades.
print("\n\t\tRESULTADOS ESPERADOS")
print("\nValor objetivo para las Ciudades:")
print(Ob)
print("Valores obtenidos al entrenar las imagenes Ciudades:")
print(Clasifica(w1, w2, En))
print("\n")
print(Clasifica(w1, w2, En_2))
print("\n")
print(Clasifica(w1, w2, En_3))
print("\n")
print(Clasifica(w1, w2, En_4))

# Mostramos los valores para Paisajes.
print("\nValor objetivo para los Paisajes:")
print(Ob2)
print("Valores obtenidos al entrenar las imagenes Paisajes:")
print(Clasifica(w1, w2, En2))
print("\n")
print(Clasifica(w1, w2, En2_2))
print("\n")
print(Clasifica(w1, w2, En2_3))
print("\n")
print(Clasifica(w1, w2, En2_4))

# Mostramos los valores de Pruebas.
print("\nValores obtenidos para las pruebas de las imagenes:")
print(Clasifica(w1, w2, img1_pr))
print("\n")
print(Clasifica(w1, w2, img2_pr))
print("\n")
print(Clasifica(w1, w2, img3_pr))
print("\n")
print(Clasifica(w1, w2, img4_pr))
print("\n")

# Datos de Ejecucion.
fin = time.time()
print("El tiempo de ejecucion es: ", (fin-inicio))
print("El error es de: ", error)

# Mostramos la Grafica.
#plt.plot(list_error)
#plt.show()