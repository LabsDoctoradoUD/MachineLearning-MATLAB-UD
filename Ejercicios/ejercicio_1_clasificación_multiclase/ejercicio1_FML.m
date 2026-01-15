%% EJERCICIO 1 - Clasificación Multiclase (Dígitos 0–9)
% Dataset: digits_multiclass.csv
% GUI: Classification Learner
%
% Flujo completo de ML:
% - Inspección y limpieza
% - Separación Train / Test
% - Normalización correcta
% - Preparación para Classification Learner
%
% Autor: Cristian Castro

clear; clc; close all;

fprintf('=== EJERCICIO 1: Clasificación Multiclase (Flujo ML Completo) ===\n\n');

%% ================== CONFIGURACIÓN ==================
USE_DEMO_DATA   = true;   % true: rápido en clase
SAMPLE_FRACTION = 0.4;    % 40% para demostración
rng(1);                   % Reproducibilidad
%% ===================================================

%% 1. Localizar ruta del script y dataset
scriptPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(scriptPath,'..','data','digits_multiclass.csv');

if ~isfile(dataPath)
    error('No se encontró digits_multiclass.csv en la carpeta data.');
end

%% 2. Cargar dataset
disp('1. Cargando dataset de dígitos...');
data = readtable(dataPath);

fprintf('Observaciones totales: %d\n\n', height(data));

%% 3. Inspección general
disp('2. Resumen del dataset:');
summary(data)
fprintf('\n');

%% 4. Reducción del dataset (modo demo)
if USE_DEMO_DATA
    disp('3. Modo DEMO activado...');
    idx = rand(height(data),1) < SAMPLE_FRACTION;
    data = data(idx,:);
    fprintf('Observaciones usadas: %d\n\n', height(data));
end

%% 5. Verificación de valores faltantes
disp('4. Verificando valores faltantes...');
missingCount = sum(ismissing(data));
disp(missingCount);

if any(missingCount > 0)
    disp('Eliminando filas con valores faltantes...');
    data = rmmissing(data);
else
    disp('No se encontraron valores faltantes.');
end
fprintf('\n');

%% 6. Separar predictores y variable objetivo
disp('5. Separando predictores y clase...');

X = data(:,1:end-1);               % pixeles
Y = categorical(data{:,end});      % dígito 0–9

fprintf('Número de variables predictoras: %d\n', width(X));
fprintf('Clases detectadas:\n');
disp(categories(Y));
fprintf('\n');

%% 7. Análisis de balance de clases
disp('6. Distribución de clases:');
tabulate(Y)
fprintf('\n');

%% 8. Separación Train / Test
disp('7. Dividiendo datos en entrenamiento y prueba (70/30)...');

cv = cvpartition(Y, 'HoldOut', 0.30);

XTrain = X(training(cv), :);
YTrain = Y(training(cv));

XTest  = X(test(cv), :);
YTest  = Y(test(cv));

fprintf('Train: %d muestras\n', height(XTrain));
fprintf('Test : %d muestras\n\n', height(XTest));

%% 9. Normalización (SIN data leakage)
disp('8. Normalizando variables (usando solo Train)...');

% Convertir a matrices numéricas
XTrainMat = XTrain{:,:};
XTestMat  = XTest{:,:};

% Calcular parámetros SOLO con train
mu    = mean(XTrainMat);
sigma = std(XTrainMat);

% Evitar división por cero (pixeles constantes)
sigma(sigma == 0) = 1;

% Normalizar
XTrainN = (XTrainMat - mu) ./ sigma;
XTestN  = (XTestMat  - mu) ./ sigma;

% Volver a tablas
XTrainN = array2table(XTrainN, 'VariableNames', XTrain.Properties.VariableNames);
XTestN  = array2table(XTestN,  'VariableNames', XTest.Properties.VariableNames);

disp('Normalización completada.');
fprintf('\n');

%% 10. Crear tabla FINAL para Classification Learner
disp('9. Preparando tabla para Classification Learner...');

tblTrain = XTrainN;
tblTrain.Digit = YTrain;

%% 11. Guardar datasets preparados
disp('10. Guardando datos preparados...');
save('datos_digits_train_test.mat', ...
     'tblTrain', ...
     'XTestN', 'YTest');

disp('Archivo datos_digits_train_test.mat guardado.');
fprintf('\n');

%% 12. Instrucciones para Classification Learner
disp('INSTRUCCIONES PARA EL EJERCICIO:');
disp('1. Se abrirá Classification Learner.');
disp('2. New Session > From Workspace.');
disp('3. Data Set Variable: tblTrain.');
disp('4. Response: Digit.');
disp('5. Validación cruzada: 5-Fold.');
disp('6. Entrenar KNN, SVM y Árbol.');
disp('7. Exportar el mejor modelo.');
disp('8. Evaluar con XTestN y YTest.');
fprintf('\n');

%% 13. Abrir Classification Learner
classificationLearner

fprintf('=== FIN DEL EJERCICIO 1 ===\n');
