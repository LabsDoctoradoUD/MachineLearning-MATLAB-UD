# Características del conjunto de datos:   # Instancias:
Multivariante                               5620

# Tipo de característica:                  # Características:
Entero                                      64


# Detalle:




Se Utilizaron programas de preprocesamiento proporcionados por el NIST para extraer mapas de bits normalizados de dígitos manuscritos de un formulario preimpreso. De un total de 43 personas, 30 contribuyeron al conjunto de entrenamiento y 13 al conjunto de prueba. 
Los mapas de bits de 32x32 se dividen en bloques no superpuestos de 4x4 y se cuenta el número de píxeles en cada bloque. Esto genera una matriz de entrada de 8x8 donde cada elemento es un entero en el rango de 0 a 16. 
Esto reduce la dimensionalidad y proporciona invariancia a pequeñas distorsiones.
