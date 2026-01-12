%% EJEMPLO 1 - Exploración del problema de Machine Learning
% Dataset: Credit Card Fraud
% Objetivo: Identificar el tipo de problema y entender los datos
% Este script es DEMOSTRATIVO (docente)

clear; clc; close all;

%% 1. Cargar el dataset
disp('Cargando dataset de fraude...');
data = readtable('../data/creditcard.csv');

%% 2. Inspección general del dataset
disp('Resumen general del dataset:');
summary(data)

fprintf('\nNúmero total de observaciones: %d\n', height(data));
fprintf('Número total de variables: %d\n', width(data));

%% 3. Identificación de la variable objetivo
% En este dataset, la variable "Class" indica:
% 0 -> Transacción normal
% 1 -> Fraude

targetVariable = 'Class';
disp(['Variable objetivo identificada: ', targetVariable]);

%% 4. Separación conceptual de X (features) y Y (target)
X = data(:, 1:end-1);
Y = categorical(data.Class);

disp('Variables predictoras (X):');
disp(X.Properties.VariableNames');

disp('Variable objetivo (Y):');
disp(categories(Y));

%% 5. Análisis de la distribución de clases
disp('Distribución de clases:');
tabulate(Y)

%% 6. Visualización del desbalance
figure;
bar(countcats(Y))
title('Distribución de Clases - Dataset de Fraude')
xlabel('Clase')
ylabel('Número de observaciones')
grid on;

%% 7. Discusión guiada (para el docente)
disp('--- DISCUSIÓN GUIADA ---');
disp('1. ¿Este es un problema de clasificación o regresión?');
disp('2. ¿Las clases están balanceadas?');
disp('3. ¿Es suficiente la accuracy como métrica?');
disp('4. ¿Qué dificultades anticipamos para entrenar un modelo?');
