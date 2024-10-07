# 🚀 **Vanguard A/B Testing: Análisis del Impacto de la Nueva Interfaz en la Finalización de Procesos**

## 💼 **Equipo de Trabajo**
- Adrián Lardiés
- Irene Sifre

## 📋 **Índice**
1. [Resumen del Proyecto](#resumen-del-proyecto)
2. [Fuentes de Datos](#fuentes-de-datos)
3. [Metodología](#metodología)
4. [Análisis Exploratorio de Datos (EDA)](#análisis-exploratorio-de-datos-eda)
5. [Rendimiento e Insights](#rendimiento-e-insights)
6. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)

---

## ✨ **Resumen del Proyecto**

### **Contexto**
Como analistas de datos en Vanguard, nuestra tarea fue evaluar la efectividad de una nueva interfaz de usuario (UI) diseñada para mejorar el compromiso de los clientes y las tasas de finalización de procesos. Esto se llevó a cabo a través de una prueba controlada A/B, comparando la nueva interfaz con la tradicional.

### **El Desafío Digital**
Vanguard identificó la necesidad de modernizar su experiencia digital para ofrecer una interfaz de usuario más intuitiva y moderna. La pregunta clave era:
*¿Lograrán estos cambios que más clientes completen sus procesos en línea con éxito?*

### **El Experimento**
El análisis se centró en el impacto de la nueva interfaz en comparación con la interfaz tradicional, evaluando su influencia en el compromiso del cliente y las tasas de finalización de procesos.

**Aspectos claves del estudio**:
- Tasas de conversión en cada paso del proceso.
- Tasas de error y abandono de usuarios.
- Diferencias en el comportamiento entre usuarios que completaron el proceso correctamente y los que no.
- Tiempos promedio empleados en cada paso tanto en el grupo de prueba como en el grupo de control.

**Periodo de estudio:** 15 de marzo de 2017 - 20 de junio de 2017

---

## 📑 **Fuentes de Datos**
- **Perfiles de Clientes**: Datos demográficos como edad, balance y nivel de actividad de los clientes.
- **Huella Digital**: Interacciones web de los usuarios, incluidos los registros de acciones en cada paso y finalización del proceso.
- **Registro del Experimento**: Información que indica si un usuario pertenecía al grupo de prueba o al grupo de control.

---

## 📚 **Metodología**

### **1. Preparación y Limpieza de Datos**
- **Unificación de Conjuntos de Datos**: Se combinaron los datos demográficos de los clientes con las interacciones digitales y el registro del experimento para crear un dataset completo.
- **Tratamiento de Valores Faltantes**: Se analizaron los valores nulos o faltantes y se aplicaron técnicas de imputación o eliminación según el caso para asegurar la integridad de los datos.

### **2. Segmentación del Viaje del Usuario**
- **Clasificación de Grupos**: Los usuarios fueron divididos en grupos de control y prueba. También se les clasificó según su rendimiento: aquellos que completaron el proceso correctamente y los que no.
- **Análisis de Rendimiento**: Se midió el tiempo tomado entre cada paso para ambos grupos, calculando promedios individuales por paso, así como promedios agregados para todo el proceso.

### **3. Visualización y Análisis de Datos**
- **Visualización**: Se utilizaron bibliotecas como Matplotlib y Seaborn para crear representaciones visuales del comportamiento de los usuarios y las métricas de rendimiento, facilitando la comparación entre los grupos.
- **Análisis Estadístico**: Se realizaron pruebas estadísticas para comparar las medias y los porcentajes de usuarios que completaron correctamente el proceso en cada paso, entre los grupos de control y prueba.

### **4. Pruebas e Implementación de Insights**
- **Pruebas de Hipótesis**: Se realizaron pruebas estadísticas para evaluar la significancia de las diferencias observadas en las métricas de rendimiento entre los dos grupos.
- **Implementación de Hallazgos**: Se desarrollaron insights accionables basados en el análisis, como la identificación de áreas donde la interfaz puede mejorar.

---

## 🔎 **Análisis Exploratorio de Datos (EDA)**

La fase de EDA se centró en comprender las métricas clave y el comportamiento del usuario a través de visualizaciones y pruebas estadísticas.

### 1. Visualizaciones
- **Diagramas de Caja**: Se visualizaron las distribuciones del tiempo empleado en cada paso del proceso para ambos grupos (prueba vs. control), identificando valores atípicos y la dispersión de los datos.
- **Histogramas**: Se evaluó la distribución de frecuencias de métricas clave, como el tiempo empleado por paso y las tasas de finalización, para analizar la asimetría de los datos.

### 2. Pruebas Estadísticas
- Se realizaron pruebas estadísticas (como t-tests o pruebas Mann-Whitney U) para comparar las medias y medianas entre los grupos de prueba y control, identificando diferencias significativas.
- Se evaluaron los tamaños del efecto para comprender la importancia práctica de los resultados, proporcionando una visión clara del impacto de la nueva interfaz.

---

## 📊 **Rendimiento e Insights**

A continuación se presentan los gráficos requeridos por el proyecto para los resultados del análisis de la prueba A/B, con un enfoque en las métricas clave como las tasas de error y finalización.

### **1. Tiempo promedio por paso**  
![image](https://github.com/user-attachments/assets/736359e2-9c73-4fed-b062-4fb4c494a072)
![image](https://github.com/user-attachments/assets/2a71d468-9ee5-4521-a021-70679833a634)
- Los tiempos son más bajos en cada paso para los clientes que completaron el proceso correctamente, excepto en el paso de Confirmación.
- En todos los pasos, la nueva versión es más eficiente que la antigua, excepto en Confirmación.
- Mejora general de la experiencia de usuario en el paso de Confirmación.

### **2. Tasas de error**  
![image](https://github.com/user-attachments/assets/030392d8-d5f3-44c3-8c28-a4ded2674e80)
- La tasa de error en el paso de Inicio es mayor en el grupo de prueba, lo que podría interpretarse como que los usuarios estaban acostumbrados al uso clásico de la web.
- El Paso 1 es notablemente deficiente en la nueva versión.
- El rendimiento en los Pasos 2 y 3 mejora en la nueva versión respecto a la tasa de error.
- Enfocarse en mejorar la experiencia de usuario en los pasos de Inicio, Paso 1 y Paso 2.

### **3. Tasas de finalización**  
![image](https://github.com/user-attachments/assets/25fab45e-6ed0-41c7-80f7-d67ab9503193)
- Tiempos más bajos en cada paso para los clientes que siguieron correctamente el proceso, excepto en el paso de Confirmación.
- En todos los pasos, la nueva versión es más eficiente que la antigua, excepto en Confirmación.
- Mejorar la experiencia de usuario en el paso de Confirmación.

---

## 📈 **Insights y Conclusiones**

- **Tasas de Finalización del Proceso**: La nueva interfaz de usuario (grupo de prueba) mostró una ligera mejora en las tasas de finalización en comparación con la interfaz tradicional (grupo de control).
- **Significancia Estadística**: Algunas diferencias entre los grupos fueron evidentes, pero no todas fueron estadísticamente significativas.
- **Segmentación de Usuarios**: Al dividir a los usuarios en aquellos que completaron el proceso correctamente y aquellos que no, se identificó que los usuarios que no siguieron los pasos correctamente fueron un punto clave de análisis.

---

## 🔎 **Recomendaciones y Limitaciones**

- **Recomendaciones**:
  - Aunque la nueva UI muestra mejoras en la eficiencia, son necesarias mejoras adicionales en varios pasos.
- **Limitaciones**:
  - Una cantidad significativa de datos (20.000 observaciones) fue rechazada por no estar correctamente clasificada como control o prueba.
  - Se recomienda realizar más pruebas incluyendo métricas o características adicionales para abordar el comportamiento de los usuarios que se desvían del proceso esperado.