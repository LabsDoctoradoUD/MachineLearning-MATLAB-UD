%% EJEMPLO 2 - Flujo completo de ML
% Clasificación con Classification Learner
% Dataset: Credit Card Fraud
%
% Objetivos:
% - Limpieza de datos
% - Análisis de clases
% - Separación Train / Test
% - Normalización correcta
% - Uso en Classification Learner
%
% Autor: Cristian Castro

clear; clc; close all;

fprintf('=== EJEMPLO 2: Clasificación - Flujo completo de ML ===\n\n');

%% ================== CONFIGURACIÓN ==================
USE_DEMO_DATA = true;     % <<< CAMBIAR SEGÚN EL CASO
SAMPLE_FRACTION = 0.4;    % 40% del dataset para demo
rng(1);                   % Reproducibilidad
%% ===================================================

%% 1. Cargar dataset
disp('1. Cargando dataset...');
data = readtable('../data/creditcard.csv');

fprintf('Dataset cargado: %d filas, %d columnas\n\n', height(data), width(data));

%% 4. Reducir dataset si es modo demo
if USE_DEMO_DATA
    fprintf('Modo DEMOSTRACIÓN activado (%.0f%% de los datos)\n', ...
        SAMPLE_FRACTION*100);

    idx = rand(height(data),1) < SAMPLE_FRACTION;
    data = data(idx,:);

    fprintf('Observaciones usadas: %d\n', height(data));
else
    fprintf('Modo COMPLETO activado\n');
end

%% 2. Inspección general
disp('2. Resumen general del dataset:');
summary(data)

%% 3. Limpieza de datos
disp('3. Verificando valores faltantes...');

missingCount = sum(ismissing(data));
disp(missingCount);

if any(missingCount > 0)
    disp('Eliminando filas con valores faltantes...');
    data = rmmissing(data);
else
    disp('No se encontraron valores faltantes.');
end
fprintf('\n');

%% 4. Separar variables predictoras y objetivo
disp('4. Separando predictores y variable objetivo...');

targetVariable = 'Class';

X = data(:, setdiff(data.Properties.VariableNames, targetVariable));
Y = categorical(data.(targetVariable));

fprintf('Número de variables predictoras: %d\n', width(X));
fprintf('Clases detectadas:\n');
disp(categories(Y));
fprintf('\n');

%% 5. Análisis de desbalance de clases
disp('5. Distribución de clases:');
tabulate(Y)
fprintf('\n');

%% 6. Separación Train / Test
disp('6. Dividiendo dataset en entrenamiento y prueba (70/30)...');

cv = cvpartition(Y, 'HoldOut', 0.30);

XTrain = X(training(cv), :);
YTrain = Y(training(cv));

XTest  = X(test(cv), :);
YTest  = Y(test(cv));

fprintf('Datos de entrenamiento: %d muestras\n', height(XTrain));
fprintf('Datos de prueba       : %d muestras\n\n', height(XTest));

%% 7. Normalización (sin data leakage)
disp('7. Normalizando variables (usando solo Train)...');

% Convertir a matrices numéricas
XTrainMat = XTrain{:,:};
XTestMat  = XTest{:,:};

% Calcular normalización SOLO con Train
mu    = mean(XTrainMat);
sigma = std(XTrainMat);

% Evitar división por cero
sigma(sigma == 0) = 1;

% Normalizar
XTrainN = (XTrainMat - mu) ./ sigma;
XTestN  = (XTestMat  - mu) ./ sigma;

% Volver a tablas
XTrainN = array2table(XTrainN, 'VariableNames', XTrain.Properties.VariableNames);
XTestN  = array2table(XTestN,  'VariableNames', XTest.Properties.VariableNames);

disp('Normalización completada.');
fprintf('\n');

%% 8. Guardar datos preparados
disp('8. Guardando datos preparados...');

save('datos_clasificacion.mat', ...
     'XTrainN', 'YTrain', ...
     'XTestN',  'YTest');

disp('Archivo "datos_clasificacion.mat" guardado correctamente.');
fprintf('\n');

%% 9. Instrucciones para Classification Learner
disp('INSTRUCCIONES PARA CLASSIFICATION LEARNER:');
disp('1. Se abrirá Classification Learner.');
disp('2. Seleccione: New Session > From Workspace.');
disp('3. Predictors: XTrainN | Response: YTrain.');
disp('4. Validación cruzada: 5-Fold.');
disp('5. Entrene y compare múltiples modelos.');
disp('6. Exporte el mejor modelo al Workspace.');
fprintf('\n');

%% 10. Abrir Classification Learner
classificationLearner

fprintf('=== FIN DEL SCRIPT ===\n');
