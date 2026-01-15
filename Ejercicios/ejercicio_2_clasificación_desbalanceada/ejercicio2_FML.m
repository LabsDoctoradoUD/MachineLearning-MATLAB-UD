%% EJERCICIO 2 - Clasificación Desbalanceada (Bank Marketing)
% Dataset: bank-additional-full.csv
% GUI: Classification Learner
%
% Flujo completo de ML:
% - Inspección y limpieza
% - Manejo de variables categóricas y numéricas
% - Separación Train / Test
% - Normalización sin data leakage
% - Evaluación con métricas adecuadas
%
% Autor: Cristian Castro

clear; clc; close all;

fprintf('=== EJERCICIO 2: Clasificación Desbalanceada (Flujo ML Completo) ===\n\n');

%% ================== CONFIGURACIÓN ==================
USE_DEMO_DATA   = true;    % true: rápido en clase
SAMPLE_FRACTION = 0.25;    % 25% para demo
rng(1);                    % Reproducibilidad
%% ===================================================

%% 1. Ruta del dataset
scriptPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(scriptPath,'..','data','bank-additional-full.csv');

if ~isfile(dataPath)
    error('No se encontró bank-additional-full.csv en la carpeta data.');
end

%% 2. Cargar dataset
disp('1. Cargando dataset Bank Marketing...');
data = readtable(dataPath, 'Delimiter',';');

fprintf('Observaciones totales: %d\n\n', height(data));

%% 3. Inspección general
disp('2. Resumen general del dataset:');
summary(data)
fprintf('\n');

%% 4. Reducción del dataset (modo demo)
if USE_DEMO_DATA
    disp('3. Modo DEMO activado...');
    idx = rand(height(data),1) < SAMPLE_FRACTION;
    data = data(idx,:);
    fprintf('Observaciones usadas: %d\n\n', height(data));
end

%% 5. Limpieza de datos
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

%% 6. Análisis del desbalance
disp('5. Distribución de la variable objetivo:');
tabulate(categorical(data.y))
fprintf('\n');

%% 7. Separar predictores y variable objetivo
disp('6. Separando predictores y clase...');

targetVariable = 'y';

X = data(:, setdiff(data.Properties.VariableNames, targetVariable));
Y = categorical(data.(targetVariable));   % yes / no

fprintf('Clases detectadas:\n');
disp(categories(Y));
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

%% 9. Normalización SOLO de variables numéricas (sin data leakage)
disp('8. Normalizando variables numéricas (usando solo Train)...');

% Identificar variables numéricas
numVars = varfun(@isnumeric, XTrain, 'OutputFormat','uniform');

% Convertir a matrices numéricas
XTrainNum = XTrain{:,numVars};
XTestNum  = XTest{:,numVars};

% Calcular parámetros SOLO con Train
mu    = mean(XTrainNum);
sigma = std(XTrainNum);
sigma(sigma == 0) = 1;

% Normalizar
XTrainNumN = (XTrainNum - mu) ./ sigma;
XTestNumN  = (XTestNum  - mu) ./ sigma;

% Reemplazar en tablas originales
XTrainN = XTrain;
XTestN  = XTest;

XTrainN(:,numVars) = array2table(XTrainNumN, ...
    'VariableNames', XTrain.Properties.VariableNames(numVars));

XTestN(:,numVars) = array2table(XTestNumN, ...
    'VariableNames', XTest.Properties.VariableNames(numVars));

disp('Normalización completada.');
fprintf('\n');

%% 10. Crear tabla FINAL para Classification Learner
disp('9. Preparando tabla para Classification Learner...');

tblTrain = XTrainN;
tblTrain.y = YTrain;

%% 11. Guardar datasets preparados
disp('10. Guardando datos preparados...');
save('datos_bank_train_test.mat', ...
     'tblTrain', ...
     'XTestN', 'YTest');

disp('Archivo datos_bank_train_test.mat guardado.');
fprintf('\n');

%% 12. Visualización dataset
figure;
uitable('Data', tblTrain(1:100,:), ...
        'Position',[20 20 1000 400]);

%% 13. Instrucciones para Classification Learner
disp('INSTRUCCIONES PARA EL EJERCICIO:');
disp('1. Se abrirá Classification Learner.');
disp('2. New Session > From Workspace.');
disp('3. Data Set Variable: tblTrain.');
disp('4. Response: y.');
disp('5. Validación cruzada: 5-Fold.');
disp('6. Entrenar Árboles, KNN y Ensemble.');
disp('7. Exportar el mejor modelo.');
disp('8. Evaluar con XTestN y YTest.');
disp('9. Analizar Recall, Precision y matriz de confusión.');
fprintf('\n');

%% 14. Abrir Classification Learner
classificationLearner

fprintf('=== FIN DEL EJERCICIO 2 ===\n');
