%% EJERCICIO 3 - Regresión (Concrete Compressive Strength)
% Dataset: Concrete_Data.xls
% GUI: Regression Learner
%
% Flujo completo de ML:
% - Inspección y limpieza
% - Separación Train / Test
% - Normalización sin data leakage
% - Evaluación con RMSE y R²
%
% Autor: Cristian Castro

clear; clc; close all;

fprintf('=== EJERCICIO 3: REGRESIÓN (Flujo ML Completo) ===\n\n');

%% ================== CONFIGURACIÓN ==================
USE_DEMO_DATA   = true;    % true: rápido en clase
SAMPLE_FRACTION = 0.4;     % 40% para demostración
rng(1);                    % Reproducibilidad
%% ===================================================

%% 1. Ruta del dataset
scriptPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(scriptPath,'..','data','Concrete_Data.xls');

if ~isfile(dataPath)
    error('No se encontró Concrete_Data.xls en la carpeta data.');
end

%% 2. Cargar dataset
disp('1. Cargando dataset de concreto...');
data = readtable(dataPath);

fprintf('Observaciones totales: %d\n', height(data));
fprintf('Variables disponibles:\n');
disp(data.Properties.VariableNames');
fprintf('\n');

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

%% 6. Separar predictores y variable objetivo
disp('5. Separando predictores y variable objetivo...');

targetVariable = 'ConcreteCompressiveStrength_MPa_Megapascals_';

X = data(:, setdiff(data.Properties.VariableNames, targetVariable));
Y = data.(targetVariable);

fprintf('Número de predictores: %d\n\n', width(X));

%% 7. Separación Train / Test
disp('6. Dividiendo datos en entrenamiento y prueba (70/30)...');

cv = cvpartition(height(data), 'HoldOut', 0.30);

XTrain = X(training(cv), :);
YTrain = Y(training(cv));

XTest  = X(test(cv), :);
YTest  = Y(test(cv));

fprintf('Train: %d muestras\n', height(XTrain));
fprintf('Test : %d muestras\n\n', height(XTest));

%% 8. Normalización (SIN data leakage)
disp('7. Normalizando variables (usando solo Train)...');

% Convertir a matrices numéricas
XTrainMat = XTrain{:,:};
XTestMat  = XTest{:,:};

% Calcular parámetros SOLO con train
mu    = mean(XTrainMat);
sigma = std(XTrainMat);
sigma(sigma == 0) = 1;

% Normalizar
XTrainN = (XTrainMat - mu) ./ sigma;
XTestN  = (XTestMat  - mu) ./ sigma;

% Volver a tablas
XTrainN = array2table(XTrainN, 'VariableNames', XTrain.Properties.VariableNames);
XTestN  = array2table(XTestN,  'VariableNames', XTest.Properties.VariableNames);

disp('Normalización completada.');
fprintf('\n');

%% 9. Crear tabla FINAL para Regression Learner
disp('8. Preparando tabla para Regression Learner...');

tblTrain = XTrainN;
tblTrain.Strength = YTrain;

%% 10. Guardar datasets preparados
disp('9. Guardando datos preparados...');
save('datos_concrete_train_test.mat', ...
     'tblTrain', ...
     'XTestN', 'YTest');

disp('Archivo datos_concrete_train_test.mat guardado.');
fprintf('\n');

%% 11. Visualización dataset
figure;
uitable('Data', tblTrain(1:100,:), ...
        'Position',[20 20 1000 400]);

%% 12. Instrucciones para Regression Learner
disp('INSTRUCCIONES PARA EL EJERCICIO:');
disp('1. Se abrirá Regression Learner.');
disp('2. New Session > From Workspace.');
disp('3. Data Set Variable: tblTrain.');
disp('4. Response: Strength.');
disp('5. Validación cruzada: 5-Fold.');
disp('6. Entrenar múltiples modelos.');
disp('7. Exportar el mejor modelo.');
disp('8. Evaluar con XTestN y YTest.');
disp('9. Analizar RMSE y R².');
fprintf('\n');

%% 13. Abrir Regression Learner
regressionLearner

fprintf('=== FIN DEL EJERCICIO 3 ===\n');
