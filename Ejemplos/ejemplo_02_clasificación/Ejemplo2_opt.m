%% EJEMPLO 2 - Clasificación con Classification Learner
% Dataset: Credit Card Fraud
% Tipo de ejemplo: Demostrativo (docente)
%
% Modo de ejecución:
% USE_DEMO_DATA = true  -> Dataset reducido (rápido, para clase)
% USE_DEMO_DATA = false -> Dataset completo (offline)

clear; clc; close all;

fprintf('=== EJEMPLO 2: Clasificación con Classification Learner ===\n');

%% ================== CONFIGURACIÓN ==================
USE_DEMO_DATA = true;     % <<< CAMBIAR SEGÚN EL CASO
SAMPLE_FRACTION = 0.1;    % 10% del dataset para demo
rng(1);                   % Reproducibilidad
%% ===================================================

%% 1. Localizar ruta del script
scriptPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(scriptPath,'..', 'data', 'creditcard.csv');

%% 2. Verificar dataset
if ~isfile(dataPath)
    error(['No se encontró creditcard.csv.\n' ...
           'Ubíquelo en la carpeta data del repositorio.']);
end

%% 3. Cargar dataset
fprintf('Cargando dataset...\n');
data = readtable(dataPath);

fprintf('Observaciones totales: %d\n', height(data));

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

%% 5. Separar variables predictoras y objetivo
targetVariable = 'Class';

X = data(:, setdiff(data.Properties.VariableNames, targetVariable));
Y = categorical(data.(targetVariable));

fprintf('Variables predictoras: %d\n', width(X));
fprintf('Clases detectadas:\n');
disp(categories(Y));

%% 6. Normalización de variables
fprintf('Normalizando variables...\n');
Xn = normalize(X);

%% 7. Guardar datos preparados para la App
savePath = fullfile(scriptPath, 'datos_clasificacion.mat');
save(savePath, 'Xn', 'Y');

fprintf('Datos preparados y guardados en:\n%s\n\n', savePath);

%% 8. Instrucciones para el instructor
disp('INSTRUCCIONES:');
disp('1. Se abrirá Classification Learner.');
disp('2. New Session > From Workspace.');
disp('3. Predictors: Xn | Response: Y.');
disp('4. Validación cruzada: 5-fold (estratificada).');
disp('5. Entrenar y comparar modelos.');

%% 9. Abrir Classification Learner
classificationLearner

fprintf('=== FIN DEL EJEMPLO 2 ===\n');
