# Jeans Mass &amp; Gravitational Collapse


Este código pretende resolver las ecuaciones de los fluidos con el término gravitatorio utilizando el método del "operator splitting".

En LWRS+gravity.py se encuentra el programa principal que se basa en resolver las ecuaciones de los fluidos utilizando un esquema de Lax-Wendroff-Ritchmyer acoplando la gravedad tras cada paso de tiempo. 

En PoissonEq.py encontramos cómo hemos resuelto la ecuación de Poisson utilizando el espacio de Fourier. Este método también aparece en el código anterior. El hecho de haber subido este .py es para poder obtener una manera más clara y directa de cómo se ha realizado la derivación de la ecuación de Poisson sin necesidad de entrar al código principal.

En datatest.py encontramos el desarrollo que hemos seguido para analizar las distintas soluciones del sistema de ecuaciones a una densidad fija y variando la presión alrededor del valor de Jeans. 

Por último, en TSN2.pdf podemos encontrar una explicación más detallada del trabajo realizado junto con un análisis de los resultados.
