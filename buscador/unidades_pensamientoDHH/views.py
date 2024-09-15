import nltk
import requests
from nltk.tokenize import word_tokenize
from django.shortcuts import render
from django.db import connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
def buscar_unidades_pensamientoDHH(parrafo, similitudUsuario):
    with connection.cursor() as cursor:
        cursor.execute("SELECT contenido, libro, capitulo, versiculos, escrito FROM unidades_pensamiento")
        resultados = cursor.fetchall()
    
    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        smooth_idf=False
    )

    contenidos = [resultado[0] for resultado in resultados]
    referencias = [(resultado[1], resultado[2], resultado[3], resultado[4]) for resultado in resultados]

    tfidf_matrix = vectorizador.fit_transform(contenidos)
    parrafo_tfidf = vectorizador.transform([parrafo])
    similitudes = cosine_similarity(parrafo_tfidf, tfidf_matrix)

    unidades_similares = []
    for similitud in sorted(similitudes[0], reverse=True):
        if similitud >= similitudUsuario:
            indice_similitud = list(similitudes[0]).index(similitud)
            unidad_pensamiento_similar = contenidos[indice_similitud]
            referencia_similar = referencias[indice_similitud]
            unidad_similar={
                'similitud': similitud,
                'contenido': unidad_pensamiento_similar,
                'libro': referencia_similar[0],
                'capitulo': referencia_similar[1],
                'versiculo': referencia_similar[2],
            }
            if len(referencia_similar) > 3 and referencia_similar[3]:
                unidad_similar['escrito'] = referencia_similar[3]
            else:
                unidad_similar['escrito'] = "Escrito no disponible"
            
            unidades_similares.append(unidad_similar)

          

    if len(unidades_similares) == 0:
        parrafo_resultados = buscar_con_Word2Vec(parrafo)
        if parrafo_resultados:
            return buscar_unidades_pensamientoDHH(parrafo_resultados, similitudUsuario)
        
    return unidades_similares

def buscar_con_Word2Vec(parrafo):
    palabras = word_tokenize(parrafo)
    resultados_lista = []
    for palabra in palabras:
        url = f'http://127.0.0.1:5000/?word={palabra}'
        response = requests.get(url)
        if response.status_code == 200:
            datos = response.json()
            for resultado in datos:
                palabra_clave = resultado[0]
                similitud = resultado[1]   
                if similitud >= 0.05:
                    resultados_lista.append(palabra_clave)
                    
    if len(resultados_lista) == 0:
        parrafo_sinonimos = buscar_y_unir_sinonimos(parrafo)
        return parrafo_sinonimos
    else:
        return ' '.join(resultados_lista)

def buscar_sinonimos(palabra):
    with connection.cursor() as cursor:
        query = "SELECT sinonimos FROM sinonimos WHERE palabra = %s"
        cursor.execute(query, (palabra,))
        resultado = cursor.fetchone()
    if resultado:
        return resultado[0]
    else:
        return None

def buscar_y_unir_sinonimos(parrafo):
    palabras = parrafo.split()
    todos_sinonimos = []
    
    for palabra in palabras:
        sinonimos = buscar_sinonimos(palabra)
        if sinonimos:
            todos_sinonimos.extend(sinonimos.split()) 
    
    if todos_sinonimos:
        return ' '.join(todos_sinonimos)
    else:
        return None

def index(request):
    return render(request, 'unidades_pensamientoDHH/index.html')

def resultados(request):
    if request.method == 'GET' and 'q' in request.GET:
        query = request.GET.get('q', '').strip()
        similitud_usuario_str = request.GET.get('similitud', '').strip()
        try:
            similitud_usuario = float(similitud_usuario_str)
        except ValueError:
            similitud_usuario = 0.05
        resultados = buscar_unidades_pensamientoDHH(query, similitud_usuario)
        if not resultados:
            return render(request, 'unidades_pensamientoDHH/notfound.html', {'query': query})
        else:
            return render(request, 'unidades_pensamientoDHH/resultados.html', {'resultados': resultados, 'query': query})
    return render(request, 'unidades_pensamientoDHH/index.html')
