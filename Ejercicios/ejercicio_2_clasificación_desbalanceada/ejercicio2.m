%% EJERCICIO 2 - Clasificación Desbalanceada (Bank Marketing)
% Dataset: bank-additional-full.csv
% GUI: Classification Learner
%
% Objetivos:
% - Analizar datasets desbalanceados
% - Evaluar métricas más allá de accuracy
% - Interpretar matrices de confusión

clear; clc; close all;

fprintf('=== EJERCICIO 2: Clasificación Desbalanceada ===\n');

%% ================== CONFIGURACIÓN ==================
USE_DEMO_DATA   = true;   % true: rápido en clase
SAMPLE_FRACTION = 0.25;   % 25% para demo
rng(1);
%% ===================================================

%% 1. Ruta del dataset
scriptPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(scriptPath, '..', 'data', 'bank-additional-full.csv');

if ~isfile(dataPath)
    error('No se encontró bank-additional-full.csv en la carpeta data.');
end

%% 2. Cargar dataset
fprintf('Cargando dataset Bank Marketing...\n');
data = readtable(dataPath, 'Delimiter',';');

fprintf('Observaciones totales: %d\n', height(data));

%% 3. Reducir dataset si es modo demo
if USE_DEMO_DATA
    idx = rand(height(data),1) < SAMPLE_FRACTION;
    data = data(idx,:);
    fprintf('Modo DEMO: %d observaciones usadas\n', height(data));
end

%% 4. Inspección del desbalance
disp('Distribución de la variable objetivo:');
disp(groupcounts(data.y));

%% 5. Separar predictores y variable objetivo
targetVariable = 'y';

X = data(:, setdiff(data.Properties.VariableNames, targetVariable));
Y = categorical(data.(targetVariable));

%% 6. Normalización SOLO de variables numéricas
numVars = varfun(@isnumeric, X, 'OutputFormat','uniform');
Xn = X;
Xn(:,numVars) = normalize(X(:,numVars));

%% 7. Crear tabla final para Classification Learner
tblClassification = Xn;
tblClassification.y = Y;

%% 8. Guardar datos preparados (opcional)
save('datos_bank.mat','tblClassification');

%% 9. Instrucciones para el ejercicio
disp('INSTRUCCIONES:');
disp('1. Se abrirá Classification Learner.');
disp('2. New Session > From Workspace.');
disp('3. Data Set Variable: tblClassification.');
disp('4. Response: y.');
disp('5. Validación cruzada (5-fold).');
disp('6. Entrenar KNN, Árbol y Ensemble.');
disp('7. Analizar matriz de confusión y Recall.');

%% 10. Abrir Classification Learner
classificationLearner

fprintf('=== FIN DEL EJERCICIO 2 ===\n');
