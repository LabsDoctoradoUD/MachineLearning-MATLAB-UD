%% EJEMPLO 3 - Regresión con Regression Learner
% Dataset: Energy Efficiency Dataset
% Tipo: Ejemplo demostrativo 
%
% Objetivos:
% - Preparar datos para regresión
% - Usar Regression Learner (GUI)
% - Comparar modelos
% - Interpretar RMSE y R²
%
% Modo:
% USE_DEMO_DATA = true  -> rápido 
% USE_DEMO_DATA = false -> completo 

clear; clc; close all;

fprintf('=== EJEMPLO 3: Regresión con Regression Learner ===\n');

%% ================== CONFIGURACIÓN ==================
USE_DEMO_DATA   = true;    % <<< cambiar según el caso
SAMPLE_FRACTION = 0.3;     % 30% para demo
rng(1);                    % Reproducibilidad
%% ===================================================

%% 1. Localizar ruta del script 
scriptPath = fileparts(mfilename('fullpath'));

% Ruta al dataset
dataPath = fullfile(scriptPath, '..', 'data', 'Energy_Efficiency.xlsx');

%% 2. Verificar existencia del dataset
if ~isfile(dataPath)
    error(['No se encontró Energy_Efficiency.xlsx.\n' ...
           'Ubique el archivo en la carpeta data del repositorio.']);
end

%% 3. Cargar dataset
fprintf('Cargando dataset de eficiencia energética...\n');
data = readtable(dataPath);

fprintf('Observaciones totales: %d\n', height(data));
disp('Variables disponibles:');
disp(data.Properties.VariableNames');

%% 4. Definir variable objetivo (regresión)
% Y1: Heating Load (valor continuo)
targetVariable = 'Y1';

if ~ismember(targetVariable, data.Properties.VariableNames)
    error('La variable objetivo Y1 no existe en el dataset.');
end

%% 5. Reducir dataset si es modo demostración
if USE_DEMO_DATA
    fprintf('Modo DEMOSTRACIÓN activado (%.0f%% de los datos)\n', ...
        SAMPLE_FRACTION*100);

    idx = rand(height(data),1) < SAMPLE_FRACTION;
    data = data(idx,:);

    fprintf('Observaciones usadas: %d\n', height(data));
else
    fprintf('Modo COMPLETO activado\n');
end

%% 6. Separar predictores (X) y objetivo (Y)
X = data(:, setdiff(data.Properties.VariableNames, targetVariable));
Y = data.(targetVariable);

fprintf('Número de variables predictoras: %d\n', width(X));

%% 7. Normalizar predictores
fprintf('Normalizando variables predictoras...\n');
Xn = normalize(X);

%% 8. Crear TABLA UNIFICADA para Regression Learner
% (esto es CLAVE para que aparezca Y1 como Response)
tblRegression = Xn;
tblRegression.Y1 = Y;

%% 9. Guardar datos preparados
savePath = fullfile(scriptPath, 'datos_regresion.mat');
save(savePath, 'tblRegression');

fprintf('Tabla de regresión guardada en:\n%s\n\n', savePath);

%% 10. Instrucciones 
disp('INSTRUCCIONES:');
disp('1. Se abrirá Regression Learner.');
disp('2. New Session > From Workspace.');
disp('3. Data Set Variable: tblRegression.');
disp('4. Response: Y1.');
disp('5. Validación cruzada: 5-fold.');
disp('6. Entrenar y comparar modelos.');

%% 11. Abrir Regression Learner App
regressionLearner

fprintf('=== FIN DEL EJEMPLO 3 ===\n');
