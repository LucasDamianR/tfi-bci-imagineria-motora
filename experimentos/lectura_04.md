## Lectura de los resultados

**1. Solo la alineación euclídea cambia el resultado.** Entre sujetos, la balanced accuracy pasa de 0,624 a 0,691 (+0,067, p Holm < 0,001), el AUC de 0,709 a 0,747 y los sujetos por encima del azar de 37 a 64 de 106. Dentro del sujeto no mejora (0,601 contra 0,604): la alineación acerca personas distintas, y dentro de una misma persona no hay nada que acercar. Con alineación, el modelo entre sujetos supera al personal en 9 puntos.

**2. La mayoría de las decisiones del pipeline importan poco.** SVM en lugar de LDA, solo mu, solo beta, la ventana 0–2 s o la 0,5–3,5 s: ninguna cambia la balanced accuracy de forma significativa. Dos empeoran en los dos esquemas: **Random Forest** (−0,024 entre sujetos), que con solo 6 características no tiene qué aprovechar y sobreajusta, y la **ventana 1–3 s** (−0,028), que pierde el tramo 0,5–1,5 s, donde el notebook 01 mostró la mayor diferencia entre manos.

**3. H2 se sostiene solo en parte.** CSP no supera a la potencia por banda en balanced accuracy (0,624 contra 0,612, p = 0,60), pero sí en AUC entre sujetos (0,709 contra 0,660, p < 0,001): ordena mejor los ensayos, pero el umbral mal ubicado (notebook 02) se come la ventaja. Algo parecido pasa con las bandas: mu sola o beta sola no cambian la balanced accuracy, pero bajan el AUC entre sujetos, así que las dos aportan información.

**4. ITR.** Con el tiempo por decisión igual al final de la ventana, la base da 2,5 bits/min entre sujetos y la alineación 4,3. La ventana 0–2 s tiene un ITR algo mayor que la base (2,55 contra 2,48 bits/min entre sujetos; 2,62 contra 2,35 dentro) porque decide medio segundo antes con casi la misma precisión.

**Limitación.** Acá el sujeto evaluado se alinea con todos sus ensayos (sin etiquetas), un supuesto optimista. El notebook 05 evalúa la versión realista, alineando solo con unos pocos ensayos de calibración.
