## Lectura de los resultados

**1. H3 se sostiene, y con margen.** Con solo 5 ensayos del sujeto nuevo, el modelo general alineado y con umbral recentrado pasa de 0,624 a 0,676 de balanced accuracy (p = 1,2 × 10⁻⁸, mejora en 78 de 106 sujetos). Con 10 ensayos llega a 0,684 y con 20 a 0,691, el mismo valor que la alineación con todos los ensayos del notebook 04. De 20 a 30 ya no mejora (p = 0,61). El ITR pasa de 2,5 a 4,3 bits/min.

**2. Lo que funciona es adaptar, no reentrenar.** Las dos estrategias que usan los k ensayos para *ajustar* el modelo general le ganan al modelo *personal* entrenado con esos mismos ensayos:

- Recentrar solo el umbral ya mejora (0,647 con 5 ensayos, sin cambiar el AUC). Confirma la lectura del notebook 02: el modelo general ordena bien, pero su punto de corte no coincide con el de cada persona.
- Alinear suma bastante más, y además sube el AUC (0,709 → 0,743): no solo corrige el umbral, sino que acerca los datos de la persona a los del entrenamiento.
- El modelo personal con 30 ensayos (0,590) queda por debajo del general sin ajustar (0,625, p = 5 × 10⁻⁴) y del personal del notebook 02 (0,604, entrenado con unos 36 ensayos).

Las dos correcciones necesitan muy pocos datos: la alineación no usa etiquetas, y el umbral solo necesita el puntaje medio de cada clase. El modelo personal, en cambio, tiene que estimar los filtros CSP y el clasificador desde cero.

**3. ¿A quién ayuda?** Con 10 ensayos, la mejora es mayor en los sujetos a los que peor les iba con el modelo general (+0,09 en el tercio inferior, +0,07 en el medio, +0,03 en el superior). Hay que leerlo con cautela: como los tercios se arman con el mismo resultado que se compara, parte de esa diferencia puede ser regresión a la media.

**Cómo leer la columna "sobre el azar".** Disminuye al aumentar k porque quedan menos ensayos de prueba y el umbral binomial sube (con 15 ensayos de prueba hace falta 0,80). Solo es comparable entre estrategias con el mismo k.

**Limitaciones.** Evaluación offline, con los ensayos de calibración sorteados del mismo registro. En una BCI real la calibración sería al comienzo de la sesión y la señal podría cambiar después. Con k grande quedan pocos ensayos de prueba por sujeto, así que cada estimación individual es ruidosa; por eso se promedian 10 sorteos.
