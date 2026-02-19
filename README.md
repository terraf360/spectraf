# spectraf

**spectraf** es un módulo de [terraf](../README.md) especializado en el procesamiento y análisis de imágenes de satélite. Proporciona una interfaz simple y poderosa para cargar, visualizar y calcular índices espectrales de diferentes sensores.

## 🎯 Características

- ✅ **Carga automática** de imágenes Landsat 9 (Level 2 Surface Reflectance)
- ✅ **Visualización intuitiva** con composiciones RGB y mapas de colores
- ✅ **Índices espectrales** preimplementados (NDVI, NDWI, EVI, SAVI)
- ✅ **API consistente** inspirada en bibliotecas científicas modernas
- ✅ **Extensible** para agregar nuevos sensores e índices

## 📦 Instalación

spectraf es parte de terraf. Asegúrate de tener las dependencias necesarias:

```bash
# Instalar con conda (recomendado)
conda env create -f environment.yml
conda activate terraf

# O con pip
pip install numpy rasterio matplotlib
```

## 🚀 Uso Rápido

```python
import spectraf

# 1. Cargar una imagen de satélite
image = spectraf.load_landsat9_image('LC09_L2SP_024048_20260110_20260111_02_T1')

# 2. Visualizar en color natural
image.show(natural_color=True)

# 3. Calcular índice de vegetación NDVI
ndvi = spectraf.calculate_ndvi(image)
ndvi.show()
```

## 📚 Documentación

### Cargar Imágenes

#### Landsat 9
```python
# Carga automática desde datos/landsat9/
image = spectraf.load_landsat9_image('LC09_L2SP_024048_20260110_20260111_02_T1')

# Especificar bandas específicas
image = spectraf.load_landsat9_image(
    'LC09_L2SP_024048_20260110_20260111_02_T1',
    bands=['B2', 'B3', 'B4', 'B5']
)
```

### Visualización

```python
# Color natural (RGB)
image.show(natural_color=True)

# Falso color (NIR-Red-Green)
image.show(bands=('B5', 'B4', 'B3'))

# Una sola banda con mapa de colores
image_single = spectraf.SatelliteImage(
    bands={'B5': image.get_band('B5')},
    metadata=image.metadata,
    sensor_type='landsat9'
)
image_single.show(cmap='RdYlGn')
```

### Índices Espectrales

#### NDVI - Índice de Vegetación
```python
ndvi = spectraf.calculate_ndvi(image)
ndvi.show()

# Valores NDVI:
#   < 0: Agua, nubes, nieve
#   0-0.2: Suelo desnudo, roca
#   0.2-0.4: Vegetación dispersa
#   > 0.4: Vegetación densa
```

#### NDWI - Índice de Agua
```python
ndwi = spectraf.calculate_ndwi(image)
ndwi.show(cmap='Blues')

# Valores NDWI:
#   > 0: Cuerpos de agua
#   < 0: No agua
```

#### EVI - Índice de Vegetación Mejorado
```python
evi = spectraf.calculate_evi(image)
evi.show()

# Menos sensible a saturación en vegetación densa
```

#### SAVI - Índice Ajustado al Suelo
```python
savi = spectraf.calculate_savi(image, L=0.5)
savi.show()

# Útil en áreas con baja cobertura vegetal
# L=0 (vegetación densa), L=1 (suelo desnudo)
```

### Clase SatelliteImage

```python
# Acceder a bandas individuales
red_band = image.get_band('B4')
nir_band = image.get_band('B5')

# Listar bandas disponibles
print(image.list_bands())  # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

# Obtener metadatos
print(image.metadata['crs'])
print(image.metadata['date'])
print(image.metadata['resolution'])

# Dimensiones
height, width = image.shape()
print(f"Imagen de {height} x {width} píxeles")
```

## 🏗️ Arquitectura

```
spectraf/
├── __init__.py          # API pública
├── core.py              # Clase SatelliteImage
├── loaders.py           # Cargadores de imágenes (Landsat 9, Sentinel-2, etc.)
├── indices.py           # Índices espectrales (NDVI, NDWI, EVI, SAVI)
├── visualization.py     # Utilidades de visualización
├── utils.py             # Funciones auxiliares
└── ejemplo_uso.py       # Ejemplo de uso completo
```

### Diseño Modular

- **core.py**: Clase `SatelliteImage` que encapsula datos y metadatos
- **loaders.py**: Funciones especializadas para cada sensor
- **indices.py**: Implementación de índices espectrales reutilizables
- **visualization.py**: Normalización y plotting separado de la lógica de negocio
- **utils.py**: Búsqueda automática de archivos y rutas

## 🔬 Índices Espectrales Soportados

| Índice | Fórmula | Uso Principal |
|--------|---------|---------------|
| **NDVI** | (NIR - Red) / (NIR + Red) | Salud y densidad de vegetación |
| **NDWI** | (Green - NIR) / (Green + NIR) | Detección de cuerpos de agua |
| **EVI** | 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1) | Vegetación densa, corrección atmosférica |
| **SAVI** | ((NIR - Red) / (NIR + Red + L)) × (1 + L) | Áreas con suelo visible |

## 🛣️ Roadmap

- [ ] Soporte para Sentinel-2
- [ ] Más índices: NDBI, NBR, MNDWI, etc.
- [ ] Exportar resultados a GeoTIFF
- [ ] Operaciones de recorte y remuestreo
- [ ] Análisis de series temporales
- [ ] Integración con Google Earth Engine

## 📝 Ejemplo Completo

Ver [ejemplo_uso.py](ejemplo_uso.py) para un ejemplo funcional completo.

```bash
# Ejecutar el ejemplo
python spectraf/ejemplo_uso.py
```

## 🤝 Contribuir

spectraf es parte del proyecto terraf. Para agregar nuevos sensores o índices:

1. **Nuevo sensor**: Agregar función en `loaders.py`
2. **Nuevo índice**: Agregar función en `indices.py` siguiendo el patrón existente
3. **Nuevas visualizaciones**: Extender `visualization.py`

## 📄 Licencia

Parte del proyecto terraf - Herramientas de procesamiento geoespacial para exploración mineral.

---

**terraf** → **spectraf** (imágenes de satélite) + otros módulos (geoquímica, magnetometría, etc.)
