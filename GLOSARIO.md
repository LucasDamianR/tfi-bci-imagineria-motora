# Glosario del TFI — BCI por imaginería motora

Versión del 8 de noviembre de 2026.

Cada término está explicado en palabras simples y, al lado, qué significa en nuestro trabajo. Los términos van en el orden en que aparecen en el pipeline: primero el cerebro y la señal, después cómo se limpia, cómo se clasifica, cómo se evalúa y cómo se comparan los resultados.

La idea en una frase: **registramos el EEG de 106 personas mientras imaginan mover la mano izquierda o la derecha, y entrenamos un modelo que adivine cuál de las dos imaginaron.** Todo lo demás de este glosario sirve para hacer eso bien o para medir qué tan bien sale.

Si aparece una palabra nueva en el código o en los notebooks, agréguenla en la tabla que corresponda.

## Cerebro y señal

| Término | Qué es | En nuestro TFI |
| --- | --- | --- |
| **BCI** (*brain-computer interface*, interfaz cerebro-computadora) | Un sistema que lee actividad del cerebro y la convierte en una orden para una máquina, sin usar músculos. | Lo que intentamos construir: un clasificador que, a partir del EEG, diga si la persona imaginó la mano izquierda o la derecha. Eso podría mover un cursor o una silla de ruedas. |
| **EEG** (electroencefalograma) | Registro de la actividad eléctrica del cerebro con electrodos sobre el cuero cabelludo. Mide voltajes muy chicos (microvoltios) y es muy ruidoso. | Dataset de PhysioNet: 64 electrodos, 160 muestras por segundo. |
| **Imaginería motora** | Imaginar un movimiento sin hacerlo. Activa la corteza motora de forma parecida al movimiento real, pero más débil. | Imaginar abrir y cerrar el puño izquierdo o el derecho (runs 4, 8 y 12). Dos clases. |
| **Corteza motora** | Franja del cerebro, arriba y al costado, que planifica y ejecuta los movimientos. Cada hemisferio controla el lado **opuesto** del cuerpo. | Los electrodos que importan son los que están encima: C3, C4 y vecinos. |
| **Sistema 10-10** | Convención internacional para ubicar y nombrar electrodos. La letra indica la zona (F frontal, C central, P parietal, O occipital); los números impares van a la izquierda, los pares a la derecha y “z” en la línea del medio. | Los 64 canales del dataset siguen este sistema. |
| **C3, C4, Cz** | Electrodos centrales del 10-10. **C3** está sobre la corteza motora izquierda (controla la mano derecha), **C4** sobre la derecha (mano izquierda), **Cz** en el medio. | Los canales que miramos en el notebook 01 para buscar el ERD. |
| **Canales motores** | Los 21 electrodos de las filas FC, C y CP, que rodean la corteza motora. | Control del notebook 02 y entrada del modelo `potencia_lda`. |
| **Ritmos cerebrales / bandas** | El EEG mezcla oscilaciones de distintas velocidades. Se agrupan por frecuencia (Hz = ciclos por segundo): delta, theta, alfa, beta, gamma. | Usamos solo mu y beta, filtrando 8–30 Hz. |
| **Ritmo mu (μ)** | Oscilación de **8–12 Hz** sobre la corteza motora. Es fuerte cuando la persona está quieta y en reposo, y **baja** cuando mueve o imagina mover. Es la misma banda que alfa, pero sobre la zona motora. | Una de las dos bandas donde buscamos la señal. Configuración `banda_mu` de la etapa 4. |
| **Ritmo beta (β)** | Oscilación de **13–30 Hz**, también ligada al sistema motor. Se comporta parecido a mu: baja durante el movimiento y rebota después. | La otra banda. Configuración `banda_beta` de la etapa 4. |
| **Potencia** | Qué tan fuerte es una oscilación (la energía de la señal en esa banda). Se suele expresar en decibeles (dB). | Lo que mide el clasificador, directa o indirectamente. |
| **ERD** (*event-related desynchronization*, desincronización relacionada con eventos) | **Caída de la potencia** en mu o beta cuando la persona mueve o imagina mover. Se llama “desincronización” porque las neuronas dejan de oscilar al unísono. Se mide como % respecto del reposo. | En el notebook 01 la potencia mu cae entre 10 % y 30 % después de la consigna, en C3 y en C4. |
| **ERS** (*event-related synchronization*) | Lo contrario: **subida** de potencia, típicamente en beta justo después de terminar un movimiento (“rebote beta”). | No lo usamos, pero aparece en la bibliografía. |
| **Contralateral / lateralización** | “Del lado opuesto”. Imaginar la mano derecha debería bajar más la potencia en C3 (izquierda), y la mano izquierda, en C4 (derecha). Esa diferencia entre hemisferios es lo que permite distinguir las dos clases. | Existe pero es chica: en C3, −31 % con la mano derecha contra −20 % con la izquierda. |
| **“Huella”** | No es un término técnico: lo usamos como sinónimo de **patrón fisiológico característico**, la marca que deja la imaginería en el EEG. En nuestro caso, la huella es el ERD contralateral en mu/beta sobre C3/C4. | Notebook 01: “¿aparece la huella?” (sí) y “¿qué tan consistente es entre sujetos?” (poco: el patrón esperado aparece en 13 de 20 sujetos en C3). |
| **Variabilidad entre sujetos** | Cada cerebro y cada cabeza son distintos: la huella cambia de lugar, de banda y de intensidad de una persona a otra. | Es la motivación del TFI: por eso un modelo entrenado con unos puede fallar con otros. |
| **Consigna (estímulo, *cue*)** | La señal en pantalla que le indica a la persona qué mano imaginar. El tiempo 0 de cada ensayo. | Anotaciones `T1` (izquierda) y `T2` (derecha) del dataset. |

## Preprocesamiento

| Término | Qué es | En nuestro TFI |
| --- | --- | --- |
| **Ensayo** (*trial*) | Una repetición de la tarea: aparece la consigna, la persona imagina unos 4 segundos, descansa. Es la unidad que se clasifica. | 4.768 ensayos en total; entre 36 y 57 por sujeto (45 en la mayoría). |
| **Época** | El pedazo de EEG recortado alrededor de un ensayo, en una ventana de tiempo fija. En la práctica, época y ensayo se usan como sinónimos. | Cada época es una matriz de 64 canales × muestras de tiempo. |
| **Ventana** | Qué tramo del ensayo se le da al modelo, en segundos desde la consigna. | Base: 0,5–2,5 s. Se descarta el primer medio segundo porque ahí domina la respuesta visual a la consigna (el pico en 0,25 s). La etapa 4 prueba 0–2, 1–3 y 0,5–3,5 s. |
| **Línea de base** | Tramo de reposo antes de la consigna, usado como referencia para medir cambios. | −1 a 0 s en el cálculo del ERD % del notebook 01. |
| **Filtro pasabanda** | Deja pasar solo las frecuencias de un rango y elimina el resto (deriva lenta, ruido de la línea eléctrica, músculo). | Base 8–30 Hz (mu + beta). |
| **Referencia promedio** (CAR, *common average reference*) | Cada electrodo mide una diferencia de voltaje contra un punto de referencia. Con CAR se le resta a cada canal el promedio de todos, lo que quita el ruido que comparten todos los electrodos. | Usada en todas las configuraciones. |
| **Frecuencia de muestreo** | Cuántas mediciones por segundo toma el equipo (Hz). | 160 Hz. Los sujetos 88, 92 y 100 se grabaron a 128 Hz y se excluyeron: por eso quedan 106 de 109. |
| **Espectro / método de Welch** | El espectro muestra cuánta potencia hay en cada frecuencia. Welch es una forma estándar de estimarlo promediando tramos cortos. | Notebook 01, gráficos de potencia en C3 y C4 por clase. |
| **Envolvente de Hilbert** | Técnica para obtener la potencia “instantánea” de una señal filtrada, muestra a muestra. | Notebook 01, curva del ERD % en el tiempo. |
| **Varianza / log-varianza** | En una señal filtrada y centrada, la varianza es su potencia. Se toma el logaritmo para que los valores sean más simétricos y le sienten mejor a un clasificador lineal. | Es la característica que sale de CSP y de `potencia_lda`. |
| **Características** (*features*) | Los números que resumen cada ensayo y que recibe el clasificador, en lugar de la señal cruda. | 6 log-varianzas de CSP, o 42 log-potencias (21 canales × 2 bandas) en `potencia_lda`. |

## Modelos

| Término | Qué es | En nuestro TFI |
| --- | --- | --- |
| **Pipeline** | La cadena completa de pasos, en orden: filtrar → recortar → extraer características → clasificar. Una “configuración” es un pipeline con elecciones concretas en cada paso. | Base: CAR, 8–30 Hz, 0,5–2,5 s, CSP + LDA. La etapa 4 cambia un paso por vez. |
| **Clasificador** | Modelo que recibe las características de un ensayo y devuelve una clase (izquierda o derecha) y, por lo general, un puntaje de confianza. | LDA, SVM o Random Forest. |
| **CSP** (*common spatial patterns*, patrones espaciales comunes) | Método que combina los 64 canales en unos pocos “canales virtuales” (filtros espaciales) elegidos para que la potencia sea **lo más distinta posible entre las dos clases**: alta con la mano izquierda y baja con la derecha, o al revés. Aprende solo dónde mirar en la cabeza, en vez de fijar C3 y C4 a mano. | 6 componentes; de cada uno se toma la log-varianza. Necesita varios ensayos para estimarse bien: con \~36 por sujeto se estima mal. |
| **Filtro espacial** | Una suma ponderada de canales (por ejemplo, 0,8·C3 − 0,5·Cz + …). Es lo que produce CSP. | — |
| **Matriz de covarianza** | Tabla de canales × canales que resume cómo varían juntos los electrodos en un ensayo. Su diagonal es la potencia de cada canal. CSP y la alineación euclídea trabajan sobre ella. | — |
| **LDA** (*linear discriminant analysis*, análisis discriminante lineal) | Clasificador lineal muy simple: busca la recta (o plano) que mejor separa las dos clases, suponiendo que cada clase forma una nube con la misma forma. Rápido y difícil de sobreajustar. | Clasificador de la configuración base (CSP + LDA, el clásico de la bibliografía). |
| **Shrinkage** (encogimiento, Ledoit-Wolf) | Corrección que “achica” una covarianza estimada con pocos datos hacia una más simple, para que no sea inestable. | En CSP (`reg="ledoit_wolf"`) y en el LDA de `potencia_lda`. |
| **SVM** (*support vector machine*, máquina de vectores de soporte) con kernel RBF | Clasificador que busca la frontera con más margen entre clases; el kernel RBF le permite que esa frontera sea curva. | Configuración `csp_svm` de la etapa 4. |
| **Random Forest** (bosque aleatorio) | Muchos árboles de decisión entrenados con partes al azar de los datos, que votan. Captura relaciones no lineales. | Configuración `csp_rf` (300 árboles). |
| **Potencia por banda + LDA** | Alternativa sin CSP: log-potencia en mu y en beta de cada uno de los 21 canales motores, directo al LDA. | Configuración `potencia_lda`: muestra si CSP aporta algo frente a lo más simple. |
| **Sobreajuste** (*overfitting*) | Cuando el modelo aprende detalles (ruido) de los datos de entrenamiento que no se repiten en datos nuevos. Rinde bien al entrenar y mal al evaluar. | Riesgo principal dentro del sujeto: pocos ensayos y muchos canales. |
| **Alineación euclídea** (EA, *Euclidean alignment*) | Transformación por sujeto que “blanquea” sus datos con la covarianza media de sus propios ensayos, para que todos los sujetos queden centrados en el mismo punto. No usa etiquetas. Reduce las diferencias entre cabezas antes de entrenar. | Configuración `alineacion_ea` (etapa 4) y estrategia `alineado_umbral` (etapa 5). He y Wu, 2020. |
| **Transferencia / modelo general** | Entrenar con muchas personas y usar el modelo en otra. Lo contrario es el modelo **personal**, entrenado solo con datos de la misma persona. | Es la comparación central del TFI. |

## Validación y métricas

| Término | Qué es | En nuestro TFI |
| --- | --- | --- |
| **Entrenamiento / prueba** | Los datos se separan: con una parte el modelo aprende y con la otra, que nunca vio, se mide qué tan bien anda. Medir con los mismos datos de entrenamiento da números inflados. | Regla de todo el trabajo. |
| **Validación cruzada** (*k-fold*) | Se parte el conjunto en k pliegues; se entrena con k−1 y se prueba con el que quedó, rotando hasta usar todos. “Estratificada”: cada pliegue mantiene la proporción de clases. “Repetida”: se hace varias veces con distintos cortes y se promedia. | Dentro del sujeto: 5 pliegues × 10 repeticiones. |
| **Dentro del sujeto** (*intra-sujeto*) | Entrenar y evaluar con ensayos de la **misma persona**. Simula una BCI calibrada a medida. | BA media 0,604. Cada modelo se entrena con \~36 ensayos. |
| **Entre sujetos / leave-subject-out** (LOSO, “dejar un sujeto afuera”) | Se entrena con todos los sujetos menos uno y se evalúa con el que quedó afuera; se repite para cada sujeto. Simula a un usuario nuevo que nunca calibró. | 106 repeticiones, cada una entrenada con 105 sujetos (\~4.700 ensayos). BA media 0,623. |
| **Comparación pareada** | Cada sujeto tiene un valor en cada esquema, así que se comparan los dos números de la misma persona en lugar de dos promedios sueltos. | Dentro vs. entre, por sujeto (test de Wilcoxon). |
| **Accuracy** (exactitud) | Porcentaje de ensayos bien clasificados. Engañosa si las clases no están balanceadas: con 70 % de una clase, decir siempre esa clase da 70 %. | La consigna pide no usarla sola. |
| **Balanced accuracy (BA)** (exactitud balanceada) | Promedio de los aciertos de cada clase: (aciertos en izquierda + aciertos en derecha) / 2. El azar da siempre 0,5, estén o no balanceadas las clases. | Métrica principal. |
| **Umbral de decisión** | El clasificador da un puntaje; por encima del umbral dice “derecha”, por debajo “izquierda”. Si el umbral está corrido, el modelo puede ordenar bien los ensayos pero equivocarse de lado. | El modelo entre sujetos tiene AUC 0,709 pero BA 0,623: buen orden, umbral mal ubicado para cada persona. |
| **AUC** (*area under the ROC curve*, área bajo la curva ROC) | Probabilidad de que, tomando al azar un ensayo de cada clase, el modelo le dé mayor puntaje al de “derecha”. Mide qué tan bien **ordena**, sin depender del umbral. 0,5 = azar, 1 = perfecto. | Métrica secundaria. La brecha AUC > BA motiva la etapa 5. |
| **Curva ROC** | Gráfico de aciertos contra falsas alarmas para todos los umbrales posibles. El AUC es el área debajo. | — |
| **Nivel de azar / umbral de azar** | Con pocos ensayos, un clasificador que adivina puede sacar más de 0,5 por suerte. El umbral de azar es la BA mínima para decir que supera al azar con p < 0,05 (prueba binomial). | Con 45 ensayos el umbral es 0,644, no 0,5. |
| **ITR** (*information transfer rate*, tasa de transferencia de información) | Cuántos bits por minuto transmite la BCI, combinando precisión y velocidad (fórmula de Wolpaw). Permite comparar BCIs distintas en una sola cifra. | Función `itr_bits_por_minuto`; para el informe. |
| **Calibración** | Los pocos ensayos que se le piden a un usuario nuevo antes de usar la BCI, para ajustar el modelo a su cabeza. | Etapa 5: con k = 0, 5, 10, 20 y 30 ensayos, ¿cuánto mejora? Estrategias: general, general\_umbral, alineado\_umbral, personal. |

## Estadística

| Término | Qué es | En nuestro TFI |
| --- | --- | --- |
| **Hipótesis** (H1, H2, H3) | Una afirmación que se escribe **antes** de mirar los resultados y que los datos pueden confirmar o no. | H1: el modelo pierde desempeño con una persona nueva. No se sostuvo. |
| **p-valor** | Probabilidad de ver una diferencia tan grande como la observada (o más) si en realidad no hubiera ninguna diferencia. Si es menor a 0,05 se dice “estadísticamente significativa”. No mide qué tan grande ni qué tan importante es la diferencia. | BA dentro vs. entre: p = 0,069 (no significativa). AUC: p = 3,9 × 10⁻⁹ (significativa). |
| **Test de Wilcoxon** (de rangos con signo, pareado) | Prueba si dos mediciones de las mismas personas difieren. Ordena las diferencias por tamaño y mira si las positivas pesan más que las negativas. No supone que los datos sigan una campana de Gauss, por eso se usa en vez del t-test cuando hay asimetrías o valores extremos. | Compara BA y AUC dentro vs. entre sujetos, sujeto por sujeto (notebook 02). |
| **Correlación de Spearman** (ρ, rho) | Mide si dos variables suben (o bajan) juntas, usando el **orden** de los valores en lugar de los valores en sí. Va de −1 a 1; 0 = sin relación. Es robusta a valores extremos y no exige relación lineal. | ρ = 0,64 entre la BA dentro y la BA entre sujetos: los sujetos que andan bien con su propio modelo también tienden a andar bien con el general. |
| **Prueba binomial** | Calcula la probabilidad de acertar k de n ensayos tirando una moneda. | Base del umbral de azar por sujeto (0,644 con 45 ensayos). |
| **Media / mediana** | La media es el promedio; la mediana es el valor del medio cuando se ordenan. La mediana no se mueve con valores extremos. | Reportamos medias por esquema y configuración. |
| **Significativo vs. relevante** | Con muchos datos, diferencias minúsculas dan p < 0,05; con pocos, diferencias grandes pueden no darlo. Hay que mirar el tamaño de la diferencia además del p-valor. | Para discutir los resultados de la etapa 4. |
| **HARKing** (*hypothesizing after the results are known*) | Mala práctica: cambiar las hipótesis después de ver los resultados para que “den”. | Por eso dejamos H1 como estaba en la propuesta aunque no se haya sostenido. |
