# ==============================================================================
# SCRIPT COMPLETO: RFSI CRUDO vs CORREGIDO (CAS + γ + PC)
# CON GRÁFICO COMPARATIVO FINAL (BIAS + KGE + POD + FAR + MORAN)
# ==============================================================================

# 0. LIBRERÍAS ----------------------------------------------------------------
paquetes <- c("sf", "dplyr", "meteo", "ranger", "terra", "lubridate", 
              "blockCV", "zoo", "parallel", "ggplot2", "spdep", "tidyr")
for (pkg in paquetes) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Directorios (ajustar según tu estructura)
dir_datos       <- "D:/S/Serbia1km/Interpolation/"
dir_salida      <- "D:/R/RFSI paper/BlockVC/Output/"
dir_raster_5km  <- "D:/R/RFSI paper/BlockVC/raster_5km/"
path_region_shp <- "D:/R/RFSI paper/clima/clasif_clima_peru/region.shp"
if (!dir.exists(dir_salida)) dir.create(dir_salida, recursive = TRUE)
options(warn = 1, stringsAsFactors = FALSE)

# 1. FUNCIÓN DE MÉTRICAS ------------------------------------------------------
calcular_metricas <- function(obs, sim, label, umbral_lluvia = 0.1) {
  idx <- complete.cases(obs, sim)
  obs <- obs[idx]; sim <- sim[idx]
  if(length(obs) == 0) {
    return(data.frame(Fold = label, N = 0, RMSE = NA, MAE = NA, R2 = NA,
                      KGE = NA, NSE = NA, BIAS = NA, POD = NA, FAR = NA, CSI = NA, ACC = NA))
  }
  rmse  <- sqrt(mean((obs - sim)^2))
  mae   <- mean(abs(obs - sim))
  r2    <- cor(obs, sim)^2
  bias  <- (sum(sim) - sum(obs)) / sum(obs) * 100
  r_cor <- cor(obs, sim)
  alpha <- sd(sim) / sd(obs)
  beta  <- mean(sim) / mean(obs)
  kge   <- 1 - sqrt((r_cor-1)^2 + (alpha-1)^2 + (beta-1)^2)
  nse   <- 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)
  
  hits <- sum(obs >= umbral_lluvia & sim >= umbral_lluvia)
  fals <- sum(obs < umbral_lluvia & sim >= umbral_lluvia)
  miss <- sum(obs >= umbral_lluvia & sim < umbral_lluvia)
  correct_neg <- sum(obs < umbral_lluvia & sim < umbral_lluvia)
  
  pod <- if(hits+miss>0) hits/(hits+miss) else NA
  far <- if(hits+fals>0) fals/(hits+fals) else NA
  csi <- if(hits+fals+miss>0) hits/(hits+fals+miss) else NA
  acc <- (hits+correct_neg)/(hits+fals+miss+correct_neg)
  
  return(data.frame(Fold = label, N = length(obs), RMSE = rmse, MAE = mae, R2 = r2,
                    KGE = kge, NSE = nse, BIAS = bias, POD = pod, FAR = far, CSI = csi, ACC = acc))
}

# 2. CARGAR DATOS Y ASIGNAR REGIÓN --------------------------------------------
cat("\n[1] Cargando datos y asignando región...")
data_raw <- read.csv(paste0(dir_datos, "RFSI7_huallaga.csv"))
zonas_clima <- st_read(path_region_shp, quiet = TRUE) %>% st_transform(32718)

data <- data_raw %>%
  mutate(date = as.Date(time, format = "%m/%d/%Y"),
         doy = yday(date), month = month(date), year = year(date), staid = sp.ID) %>%
  filter(!is.na(prcp), !is.na(precland), !is.na(dem), !is.na(tmax), !is.na(tmin), prcp >= 0) %>%
  distinct()

data_sf <- st_as_sf(data, coords = c("lon","lat"), crs = 4326) %>% st_transform(32718)
coords_utm <- st_coordinates(data_sf)
data <- data %>% mutate(x_utm = coords_utm[,"X"], y_utm = coords_utm[,"Y"])

data_sf_trabajo <- st_as_sf(data, coords = c("x_utm","y_utm"), crs = 32718, remove = FALSE)
data_con_zona <- st_join(data_sf_trabajo, zonas_clima["Region"], join = st_intersects)
data <- data_con_zona %>%
  as.data.frame() %>%
  dplyr::select(-geometry) %>%
  mutate(doy_sin = sin(2*pi*doy/365.25),
         doy_cos = cos(2*pi*doy/365.25),
         z = as.numeric(date - min(date))) %>%
  filter(!is.na(Region))
cat("Regiones: ", paste(unique(data$Region), collapse = ", "), "\n")

# 3. GENERACIÓN DE BLOQUES ESPACIALES (blockCV) ---------------------------------
cat("\n[2] Generando folds espaciales (blockCV)...")
pa_data <- data %>%
  group_by(staid, x_utm, y_utm) %>% slice(1) %>% ungroup() %>%
  st_as_sf(coords = c("x_utm","y_utm"), crs = 32718)

r_base <- rast(c(paste0(dir_raster_5km, "dem_5km.tif"),
                 paste0(dir_raster_5km, "slope_5km.tif"),
                 paste0(dir_raster_5km, "sin_5km.tif"),
                 paste0(dir_raster_5km, "cos_5km.tif"),
                 paste0(dir_raster_5km, "twi_5km.tif"),
                 paste0(dir_raster_5km, "dem_sd_5km.tif")))
names(r_base) <- c("demav","slope","sin","cos","twi","demsd")

set.seed(42)
sb2 <- cv_spatial(x = pa_data, r = r_base, k = 5, size = 156268, hexagon = TRUE,
                  selection = "random", iteration = 100, biomod2 = TRUE, raster_colors = terrain.colors(10, rev = TRUE))

cv_plot(cv = sb2, 
        r = r_base, #r_clima[[2]]
        x = pa_data, raster_colors = terrain.colors(10, alpha = 0.5))+
theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7.5))


# cv_spatial_autocor(
#   r = r_base, # a SpatRaster object or path to files
#   num_sample = 5000, # number of cells to be used
#   plot = TRUE, raster_colors = terrain.colors(10, alpha = 0.5)
# )

# cv_similarity(cv = sb2, # the environmental clustering
#               x = pa_data, 
#               r = r_base, 
#               method = "MESS",
#               progress = FALSE)

folds_ref <- data.frame(staid = pa_data$staid, fold_spatial = sb2$folds_ids)
data <- data %>% left_join(folds_ref, by = "staid")
if(length(unique(data$fold_spatial)) != 5) warning("Número de folds distinto de 5")
cat("Distribución de estaciones por fold y región:\n")
print(table(data$Region, data$fold_spatial))

# 4. SELECCIÓN DE PREDICTORES (Spearman) ----------------------------------------
cat("\n[3] Seleccionando predictores significativos (α = 0.01)...")
predictores_candidatos <- c("precland", "tmax", "tmin", "demav", "demsd", "twi", "slope", "sin", "cos")
predictores_validos <- c()
for (pred in predictores_candidatos) {
  if(!pred %in% names(data)) next
  df_test <- data %>% filter(!is.na(!!sym(pred)), !is.na(prcp))
  prueba <- cor.test(df_test[[pred]], df_test$prcp, method = "spearman")
  if (!is.na(prueba$p.value) && prueba$p.value < 0.01) {
    predictores_validos <- c(predictores_validos, pred)
  }
}
predictores_finales <- unique(c(predictores_validos, "doy_sin", "doy_cos"))
fm.RFSI <- as.formula(paste("prcp ~", paste(predictores_finales, collapse = " + ")))
cat("Fórmula final:\n"); print(fm.RFSI)

# 5. VALIDACIÓN CRUZADA PARA AMBOS MODELOS --------------------------------------
cat("\n[4] Iniciando validación cruzada (crudo + corregido simultáneamente)...")
pred_raw <- data.frame()   # predicciones sin corrección (columna pred_rfsi)
pred_cor <- data.frame()   # predicciones corregidas (columna pred_pc)
res_cv_raw <- data.frame()
res_cv_cor <- data.frame()
total_estaciones <- n_distinct(data$staid)

for (f in 1:5) {
  cat(sprintf("\n>>> Procesando Fold %d/5... ", f))
  train_cv <- as.data.frame(data[data$fold_spatial != f, ])
  test_cv  <- as.data.frame(data[data$fold_spatial == f, ]) %>% dplyr::filter(!is.na(prcp))
  
  # Modelo RFSI base (entrenado una sola vez)
  mod_rfsi <- rfsi(formula = fm.RFSI, data = train_cv,
                   data.staid.x.y.z = c("staid", "x_utm", "y_utm", "z"),
                   n.obs = 5, cpus = max(1, parallel::detectCores() - 2), num.trees = 250)
  
  pred_train <- meteo::pred.rfsi(mod_rfsi, train_cv, "prcp",
                                 c("staid","x_utm","y_utm","z"),
                                 newdata = train_cv,
                                 newdata.staid.x.y.z = c("staid","x_utm","y_utm","z"))
  pred_test  <- meteo::pred.rfsi(mod_rfsi, train_cv, "prcp",
                                 c("staid","x_utm","y_utm","z"),
                                 newdata = test_cv,
                                 newdata.staid.x.y.z = c("staid","x_utm","y_utm","z"))
  
  train_cv$pred_rfsi <- pmax(0, pred_train$pred)
  test_cv$pred_rfsi  <- pmax(0, pred_test$pred)
  
  # --- Modelo CRUDO: guardar predicciones y métricas ---
  pred_raw <- rbind(pred_raw,
                    test_cv %>% dplyr::select(staid, date, month, year, Region, prcp) %>%
                      mutate(pred_rfsi = test_cv$pred_rfsi, fold_spatial = f))
  res_cv_raw <- rbind(res_cv_raw, calcular_metricas(test_cv$prcp, test_cv$pred_rfsi, paste0("Fold ", f)))
  
  # --- Modelo CORREGIDO: aplicar CAS + γ + PC ---
  # Umbral mensual p99
  umbral_mensual_train <- train_cv %>%
    dplyr::filter(prcp > 0) %>%
    dplyr::group_by(Region, month) %>%
    dplyr::summarise(p99 = quantile(prcp, 0.99, na.rm = TRUE), .groups = 'drop')
  
  # Factores CAS (0.6-1.8) suavizados
  factors_train <- train_cv %>%
    dplyr::group_by(Region, year, month, staid) %>%
    dplyr::summarise(obs_mes = sum(prcp), pred_mes = sum(pred_rfsi), .groups = "drop") %>%
    dplyr::mutate(factor_estacion = ifelse(pred_mes > 0, obs_mes / pred_mes, 1.0)) %>%
    dplyr::group_by(Region, year, month) %>%
    dplyr::summarise(factor_regional = mean(factor_estacion, na.rm = TRUE),
                     n_est = n(), .groups = "drop") %>%
    dplyr::mutate(
      factor_regional = ifelse(n_est < 3, 1.0, factor_regional),
      factor_regional = pmax(0.6, pmin(1.8, factor_regional)),
      factor_suavizado = zoo::rollapply(factor_regional, width = 3, FUN = mean,
                                        fill = factor_regional, align = "center",partial = TRUE)
    )
  
  # Estadísticas mensuales para corrección de varianza (γ)
  stats_train <- train_cv %>%
    dplyr::group_by(Region, month) %>%
    dplyr::summarise(
      mean_obs  = mean(prcp, na.rm = TRUE),
      sd_obs    = sd(prcp, na.rm = TRUE),
      mean_pred = mean(pred_rfsi, na.rm = TRUE),
      sd_pred   = sd(pred_rfsi, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(gamma = pmax(0.5, pmin(2.0, sd_obs / sd_pred)))
  
  # ==========================================================================
  # CAMBIO: Estadísticas diarias con umbral dinámico POR REGIÓN
  # ==========================================================================
  estadisticas_diarias <- train_cv %>%
    dplyr::group_by(Region, date) %>%
    dplyr::summarise(
      estaciones_con_lluvia = sum(prcp > 0.1, na.rm = TRUE),
      total_estaciones_reg = n(),   # número de estaciones con dato ese día en esa región
      umbral_inferior = max(0.5, quantile(prcp, 0.15, na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      umbral_dinamico_reg = pmax(2, round(0.10 * total_estaciones_reg))
    )
  
  # Aplicar correcciones a test (CAS + γ + PC) usando umbral regional
  test_cv_cor <- test_cv %>%
    dplyr::left_join(factors_train %>% dplyr::select(Region, year, month, factor_suavizado), 
                     by = c("Region", "year", "month")) %>%
    dplyr::left_join(stats_train %>% dplyr::select(Region, month, mean_obs, mean_pred, gamma), 
                     by = c("Region", "month")) %>%
    dplyr::left_join(estadisticas_diarias, by = c("Region", "date")) %>%
    dplyr::left_join(umbral_mensual_train, by = c("Region", "month")) %>%
    dplyr::mutate(
      factor_suavizado = ifelse(is.na(factor_suavizado), 1.0, factor_suavizado),
      pred_cas = pred_rfsi * factor_suavizado,
      mean_obs = ifelse(is.na(mean_obs), 0, mean_obs),
      mean_pred = ifelse(is.na(mean_pred), 0, mean_pred),
      gamma = ifelse(is.na(gamma), 1.0, gamma),
      pred_var = mean_obs + gamma * (pred_cas - mean_pred),
      pred_var = pmax(0, pred_var),
      p99 = ifelse(is.na(p99), Inf, p99),
      pred_pc = pred_var,
      # PC con umbral dinámico regional
      pred_pc = ifelse(estaciones_con_lluvia < umbral_dinamico_reg, pred_pc * 0.6, pred_pc),
      pred_pc = ifelse(estaciones_con_lluvia >= umbral_dinamico_reg & pred_pc < umbral_inferior, 0, pred_pc),
      pred_pc = ifelse(pred_pc > p99, p99, pred_pc),
      pred_pc = ifelse(total_estaciones_reg > 0 & estaciones_con_lluvia == 0, 0, pred_pc),
      pred_pc = pmax(0, pred_pc)
    )
  
  pred_cor <- rbind(pred_cor,
                    test_cv_cor %>% dplyr::select(staid, date, month, year, Region, prcp, pred_pc) %>%
                      mutate(fold_spatial = f))
  res_cv_cor <- rbind(res_cv_cor, calcular_metricas(test_cv_cor$prcp, test_cv_cor$pred_pc, paste0("Fold ", f)))
  
  cat("Completado.")
  gc()
}

# 6. RESULTADOS MÉTRICAS (opcional, para ver en consola) ----------------------
cat("\n\n======== MÉTRICAS MODELO CRUDO ========\n")
final_raw <- rbind(res_cv_raw, res_cv_raw %>% summarise(Fold = "AVERAGE", across(N:ACC, mean, na.rm = TRUE)))
print(final_raw)

cat("\n\n======== MÉTRICAS MODELO CORREGIDO ========\n")
final_cor <- rbind(res_cv_cor, res_cv_cor %>% summarise(Fold = "AVERAGE", across(N:ACC, mean, na.rm = TRUE)))
print(final_cor)

# ==============================================================================
# 7. GRÁFICO COMPARATIVO ÚNICO (CRUDO vs CORREGIDO) - IGUAL AL QUE SOLICITAS
# ==============================================================================
cat("\n[5] Generando gráfico comparativo final (Raw vs Corrected)...")

# Calcular Moran's I para ambos modelos
pred_raw <- pred_raw %>% mutate(residuo = prcp - pred_rfsi)
res_est_raw <- pred_raw %>% group_by(staid) %>% summarise(res_medio = mean(residuo, na.rm=TRUE), .groups="drop")
coords_estacion <- data %>% distinct(staid, x_utm, y_utm)
res_est_raw <- res_est_raw %>% left_join(coords_estacion, by="staid") %>% filter(complete.cases(x_utm, y_utm, res_medio))
coords_mat <- as.matrix(res_est_raw[, c("x_utm","y_utm")])
nb <- knn2nb(knearneigh(coords_mat, k=5))
lw <- nb2listw(nb, style="W", zero.policy=TRUE)
moran_raw <- moran.test(res_est_raw$res_medio, lw, zero.policy=TRUE)

pred_cor <- pred_cor %>% mutate(residuo = prcp - pred_pc)
res_est_cor <- pred_cor %>% group_by(staid) %>% summarise(res_medio = mean(residuo, na.rm=TRUE), .groups="drop") %>% 
  left_join(coords_estacion, by="staid") %>% filter(complete.cases(x_utm, y_utm, res_medio))
moran_cor <- moran.test(res_est_cor$res_medio, lw, zero.policy=TRUE)

# Extraer distribución de estaciones por fold y región
estaciones_folds <- data %>% distinct(staid, Region, fold_spatial)
todas_combinaciones <- expand.grid(fold_spatial = 1:5, Region = unique(data$Region), stringsAsFactors = FALSE)

metadatos <- todas_combinaciones %>% 
  left_join(estaciones_folds %>% group_by(fold_spatial, Region) %>% summarise(Est_Test = n(), .groups="drop"), by=c("fold_spatial","Region")) %>%
  mutate(Est_Test = ifelse(is.na(Est_Test), 0, Est_Test)) %>%
  left_join(estaciones_folds %>% distinct(staid,Region) %>% group_by(Region) %>% summarise(Total_Est_Reg = n(), .groups="drop"), by="Region") %>%
  mutate(Est_Train = Total_Est_Reg - Est_Test)

# Métricas por fold/región para crudo
metrics_raw <- pred_raw %>%
  group_by(fold_spatial, Region) %>%
  summarise(
    BIAS = (sum(pred_rfsi,na.rm=T)-sum(prcp,na.rm=T))/sum(prcp,na.rm=T)*100,
    KGE = { sim<-pred_rfsi; obs<-prcp; if(sum(!is.na(sim))>5){ r<-cor(sim,obs,use="complete.obs"); a<-sd(sim,na.rm=T)/sd(obs,na.rm=T); b<-mean(sim,na.rm=T)/mean(obs,na.rm=T); 1-sqrt((r-1)^2+(a-1)^2+(b-1)^2) } else {NA} },
    POD = sum(pred_rfsi>0.1 & prcp>0.1,na.rm=T)/sum(prcp>0.1,na.rm=T),
    FAR = sum(pred_rfsi>0.1 & prcp<=0.1,na.rm=T)/sum(pred_rfsi>0.1,na.rm=T),
    .groups="drop"
  ) %>% mutate(Modelo = "Raw RFSI")

# Métricas por fold/región para corregido
metrics_cor <- pred_cor %>%
  group_by(fold_spatial, Region) %>%
  summarise(
    BIAS = (sum(pred_pc,na.rm=T)-sum(prcp,na.rm=T))/sum(prcp,na.rm=T)*100,
    KGE = { sim<-pred_pc; obs<-prcp; if(sum(!is.na(sim))>5){ r<-cor(sim,obs,use="complete.obs"); a<-sd(sim,na.rm=T)/sd(obs,na.rm=T); b<-mean(sim,na.rm=T)/mean(obs,na.rm=T); 1-sqrt((r-1)^2+(a-1)^2+(b-1)^2) } else {NA} },
    POD = sum(pred_pc>0.1 & prcp>0.1,na.rm=T)/sum(prcp>0.1,na.rm=T),
    FAR = sum(pred_pc>0.1 & prcp<=0.1,na.rm=T)/sum(pred_pc>0.1,na.rm=T),
    .groups="drop"
  ) %>% mutate(Modelo = "PC + CAS + Var Corrig.")

# Consolidar
df_comp <- bind_rows(metrics_raw, metrics_cor) %>%
  left_join(metadatos, by = c("fold_spatial","Region"))

# Preparar etiquetas y posiciones
alturas_max <- df_comp %>% group_by(fold_spatial) %>% summarise(Max_BIAS = max(c(BIAS,0), na.rm=T), .groups="drop")
tabla_plot <- df_comp %>%
  left_join(alturas_max, by="fold_spatial") %>%
  mutate(
    Es_Vacio = (Est_Test == 0),
    Region_Label = sprintf("%s\n(N=%d)", Region, Total_Est_Reg),
    Fold_Label = sprintf("Spatial Fold %d", fold_spatial),
    Label_Metricas = ifelse(Es_Vacio, NA_character_,
                            sprintf("Tr: %d | Ts: %d\nKGE: %.2f\nPOD: %.2f\nFAR: %.2f", Est_Train, Est_Test, KGE, POD, FAR)),
    Label_No_Data = ifelse(Es_Vacio, sprintf("N/A\n(Tr: %d | Ts: 0)\nNo Test Data", Est_Train), NA_character_),
    Y_Text = ifelse(Modelo == "Raw RFSI", Max_BIAS + 85, Max_BIAS + 35)
  )

subtitulo <- sprintf(
  "Spatial Moran's I (Residuals)  •  Raw Model: %.3f (p: %.3f)   vs   Corrected Model: %.3f (p: %.3f)",
  moran_raw$estimate[1], moran_raw$p.value, moran_cor$estimate[1], moran_cor$p.value
)

# Gráfico final
p_comparativo <- ggplot(tabla_plot, aes(x = Region_Label, y = BIAS, fill = Modelo)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75), width = 0.65, color = "grey30", linewidth = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  scale_fill_manual(values = c("Raw RFSI" = "#9CA3AF", "PC + CAS + Var Corrig." = "#10B981")) +
  geom_text(aes(label = sprintf("%+.1f%%", BIAS), vjust = ifelse(BIAS > 0, -0.4, 1.3)),
            position = position_dodge(width = 0.75), size = 2.9, fontface = "bold") +
  geom_text(data = subset(tabla_plot, !Es_Vacio), aes(y = Y_Text, label = Label_Metricas, color = Modelo),
            position = position_dodge(width = 0.75), size = 2.3, fontface = "bold", lineheight = 0.9) +
  geom_text(data = subset(tabla_plot, Es_Vacio & Modelo == "Raw RFSI"), aes(x = Region_Label, y = 60, label = Label_No_Data),
            size = 2.5, color = "grey40", fontface = "italic", lineheight = 0.95) +
  scale_color_manual(values = c("Raw RFSI" = "#374151", "PC + CAS + Var Corrig." = "#047857")) +
  facet_wrap(~Fold_Label, nrow = 1, scales = "free_y") +
  theme_bw(base_size = 11) +
  labs(
    title = "Spatial Cross-Validation Diagnostic Anatomy: Raw vs. Physically Constrained RFSI",
    subtitle = subtitulo,
    x = "Physiographic Region (Total Regional Stations)",
    y = "Volumetric BIAS (%)",
    fill = "Model Scenario:"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
    plot.subtitle = element_text(size = 9, face = "bold.italic", color = "#111827", hjust = 0.5),
    strip.background = element_rect(fill = "grey95", color = "grey70"),
    strip.text = element_text(face = "bold", size = 9.5),
    axis.text.x = element_text(face = "bold", size = 8.5),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 9),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  guides(color = "none")

print(p_comparativo)
#ggsave(paste0(dir_salida, "Comparativo_Raw_vs_Corrected.png"), p_comparativo, width = 14, height = 6, dpi = 300)



## ==============================================================================
# 2. PROCESAMIENTO Y AGREGACIÓN DE MÉTRICAS COMPARATIVAS POR REGIÓN Y FOLD
# ==============================================================================
cat("\n[2] Extrayendo métricas regionales comparativas (KGE, POD, FAR)...")

# Unimos de manera segura las predicciones de ambos modelos bajo una estructura común
predicciones_totales <- pred_raw %>%
  dplyr::inner_join(pred_cor %>% dplyr::select(staid, date, fold_spatial, pred_pc), 
                    by = c("staid", "date", "fold_spatial"))

# Metadatos dinámicos de estaciones
estaciones_folds <- data %>% dplyr::distinct(staid, Region, fold_spatial)

todas_combinaciones <- expand.grid(
  fold_spatial = sort(unique(data$fold_spatial)),
  Region = unique(data$Region),
  stringsAsFactors = FALSE
)

estaciones_test <- todas_combinaciones %>%
  dplyr::left_join(
    estaciones_folds %>%
      dplyr::group_by(fold_spatial, Region) %>% 
      dplyr::summarise(Est_Test = dplyr::n(), .groups = "drop"),
    by = c("fold_spatial", "Region")
  ) %>%
  dplyr::mutate(Est_Test = ifelse(is.na(Est_Test), 0, Est_Test))

estaciones_totales_reg <- estaciones_folds %>%
  dplyr::distinct(staid, Region) %>%
  dplyr::group_by(Region) %>%
  dplyr::summarise(Total_Est_Reg = dplyr::n(), .groups = "drop")

metadatos_dinamicos <- estaciones_test %>%
  dplyr::left_join(estaciones_totales_reg, by = "Region") %>%
  dplyr::mutate(Est_Train = Total_Est_Reg - Est_Test) %>%
  dplyr::select(fold_spatial, Region, Total_Est_Reg, Est_Train, Est_Test)

# Función interna matemática para evitar duplicidad al evaluar escenarios
calcular_bloque_metricas <- function(df, columna_pred) {
  df %>%
    dplyr::group_by(fold_spatial, Region) %>%
    dplyr::summarise(
      BIAS = (sum(!!sym(columna_pred), na.rm = TRUE) - sum(prcp, na.rm = TRUE)) / sum(prcp, na.rm = TRUE) * 100,
      KGE = {
        sim <- !!sym(columna_pred)
        obs <- prcp
        if(sum(!is.na(sim)) > 5) {
          r <- cor(sim, obs, use = "complete.obs")
          alpha <- sd(sim, na.rm=TRUE) / sd(obs, na.rm=TRUE)
          beta  <- mean(sim, na.rm=TRUE) / mean(obs, na.rm=TRUE)
          1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
        } else { NA_real_ }
      },
      POD = sum(!!sym(columna_pred) > 0.1 & prcp > 0.1, na.rm = TRUE) / sum(prcp > 0.1, na.rm = TRUE),
      FAR = sum(!!sym(columna_pred) > 0.1 & prcp <= 0.1, na.rm = TRUE) / sum(!!sym(columna_pred) > 0.1, na.rm = TRUE),
      .groups = 'drop'
    )
}

metricas_raw <- calcular_bloque_metricas(predicciones_totales, "pred_rfsi") %>% dplyr::mutate(Model = "Raw RFSI")
metricas_cor <- calcular_bloque_metricas(predicciones_totales, "pred_pc")   %>% dplyr::mutate(Model = "PC + CAS + Var Corrig.")

# Consolidamos ambos modelos en formato largo para ggplot
df_metricas_long <- rbind(metricas_raw, metricas_cor)

# ==============================================================================
# 3. CONSTRUCCIÓN DEL COMPUESTO GRÁFICO CIENTÍFICO MULTI-ESCENARIO
# ==============================================================================
# ==============================================================================
# 3. PREPARACIÓN DE ETIQUETAS CON TEXTO REDISEÑADO Y APILAMIENTO
# ==============================================================================
cat("\n[3] Estructurando etiquetas y renderizando panel comparativo de alta resolución...")

# Asegurar que todas las combinaciones de Fold y Región existan en los metadatos
metadatos_completos <- metadatos %>%
  dplyr::select(fold_spatial, Region, Total_Est_Reg, Est_Train, Est_Test)

# Combinar métricas e inyectar las regiones vacías de forma explícita
tabla_plot <- df_comp %>%
  dplyr::select(-Total_Est_Reg, -Est_Train, -Est_Test) %>% 
  dplyr::right_join(metadatos_completos, by = c("fold_spatial", "Region")) %>%
  dplyr::mutate(
    Es_Vacio = (Est_Test == 0),
    Region_Label = sprintf("%s\n(N=%d)", Region, Total_Est_Reg),
    Fold_Label = sprintf("Spatial Fold %d", fold_spatial),
    
    # Forzar el nombre del modelo para mantener la estructura estética
    Modelo = ifelse(is.na(Modelo), "Raw RFSI", Modelo),
    
    # Bloque estadístico unificado por barra individual (Capa B)
    Label_Metricas = ifelse(Es_Vacio, NA_character_,
                            sprintf("Tr: %d | Ts: %d\nKGE: %.2f\nPOD: %.2f\nFAR: %.2f", 
                                    Est_Train, Est_Test, KGE, POD, FAR))
  )

# Calcular la altura máxima por Fold para ubicar las leyendas aéreas sin colisiones
alturas_max <- tabla_plot %>% 
  dplyr::group_by(fold_spatial) %>% 
  dplyr::summarise(Max_BIAS = max(c(BIAS, 0), na.rm = TRUE), .groups = "drop")

# Asignar coordenadas 'Y' bien separadas en vertical aprovechando el espacio aéreo
tabla_plot <- tabla_plot %>%
  dplyr::left_join(alturas_max, by = "fold_spatial") %>%
  dplyr::mutate(
    # Espaciado vertical optimizado (Piso 2 y Piso 1)
    Y_Techo_Metricas = ifelse(Modelo == "Raw RFSI", 
                              Max_BIAS + pmax(45, Max_BIAS * 0.55),   
                              Max_BIAS + pmax(15, Max_BIAS * 0.20))   
  )

# Aislamiento de las regiones vacías (Capa C)
tabla_vacio_modelo <- tabla_plot %>%
  dplyr::filter(Es_Vacio) %>%
  dplyr::distinct(fold_spatial, Region_Label, Fold_Label, Est_Train, Max_BIAS) %>%
  dplyr::mutate(
    Label_No_Data = sprintf("Tr: %d | Ts: 0\n(N/A)\nNo Test Data", Est_Train),
    Y_Vacio = pmax(25, Max_BIAS * 0.35) 
  )

# Subtitulado basado en la autocorrelación espacial calculada
subtitulo <- sprintf(
  "Spatial Moran's I (Residuals)  •  Raw Model: %.3f (p: %.3f)   vs   Corrected Model: %.3f (p: %.3f)",
  moran_raw$estimate[1], moran_raw$p.value, moran_cor$estimate[1], moran_cor$p.value
)

# ==============================================================================
# 4. RENDERIZADO DEL LIENZO CON LETRA GRANDE (OPTIMIZADO)
# ==============================================================================
p_comparativo <- ggplot(tabla_plot, aes(x = Region_Label, y = BIAS, fill = Modelo)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75), width = 0.65, color = "grey30", linewidth = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  
  scale_fill_manual(values = c("Raw RFSI" = "#9CA3AF", "PC + CAS + Var Corrig." = "#10B981")) +
  scale_color_manual(values = c("Raw RFSI" = "#374151", "PC + CAS + Var Corrig." = "#047857")) +
  
  # CAPA A: Porcentaje de BIAS en la punta de la barra (Aumentado a size = 3.5)
  geom_text(data = subset(tabla_plot, !Es_Vacio), 
            aes(label = sprintf("%+.1f%%", BIAS), vjust = ifelse(BIAS > 0, -0.4, 1.3)),
            position = position_dodge(width = 0.75), size = 2.5, fontface = "bold") +
  
  # CAPA B: Métricas en el techo separadas verticalmente (Aumentado a size = 2.8)
  geom_text(data = subset(tabla_plot, !Es_Vacio), 
            aes(y = Y_Techo_Metricas, label = Label_Metricas, color = Modelo),
            position = position_dodge(width = 0.75), size = 2.5, fontface = "bold", lineheight = 0.90, vjust = 0.5) +
  
  # CAPA C: Folds vacíos (Ts = 0) (Aumentado a size = 3.0)
  geom_text(data = tabla_vacio_modelo, aes(
    x = Region_Label, 
    y = Y_Vacio, 
    label = Label_No_Data
  ), inherit.aes = FALSE, size = 2.0, color = "grey40", fontface = "italic", lineheight = 0.95, vjust = 0.5) +
  
  facet_wrap(~Fold_Label, nrow = 1, scales = "free_y") +
  
  # INCREMENTO GENERAL DE LÍNEAS DE TEXTO DEL TEMA (base_size = 13)
  theme_bw(base_size = 11) +
  labs(
    title = "Spatial Cross-Validation Diagnostic Anatomy: Raw vs. Physically Constrained RFSI",
    subtitle = subtitulo,
    x = "Physiographic Region (Total Regional Stations)",
    y = "Volumetric BIAS (%)",
    fill = "Model Scenario:"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
    plot.subtitle = element_text(size = 9.5, face = "bold.italic", color = "black", hjust = 0.5),
    strip.background = element_rect(fill = "grey93", color = "grey70"),
    strip.text = element_text(face = "bold", size = 9),
    axis.text.x = element_text(face = "bold", size = 8),
    axis.text.y = element_text(size = 9),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 9),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  guides(color = "none")

# Desplegar en la consola gráfica de R
print(p_comparativo)

# ==============================================================================
# 5. EXPORTACIÓN FÍSICA COMPACTA (EL SECRETO DEL CAMBIO DE ESCALA)
# ==============================================================================
# Al forzar dimensiones físicas externas compactas (ej: 11 x 5.5 pulgadas), 
# las fuentes internas configuradas con tamaños grandes se verán gigantes y nítidas.
# ggsave(
#   filename = "spatial_validation_compact_large_text.png", 
#   plot = p_comparativo, 
#   width = 11.0, 
#   height = 5.5, 
#   dpi = 300, 
#   units = "in"
# )





# ==============================================================================
# GRÁFICO COMPLEMENTARIO: CICLO ANUAL (OBSERVADO vs PREDICHO CORREGIDO)
# ==============================================================================
cat("\n[+] Generando gráfico de validación estacional (modelo corregido)...")

# ==============================================================================
# GRÁFICO COMPARATIVO DEL CICLO ANUAL: OBSERVADO vs CRUDO vs CORREGIDO
# ==============================================================================
cat("\n[+] Generando gráfico estacional combinado (Observado + Crudo + Corregido)...")

# 1. Preparar perfil del modelo crudo
perfil_crudo <- pred_raw %>%
  dplyr::group_by(Region, month) %>%
  dplyr::summarise(Precipitacion = mean(pred_rfsi, na.rm = TRUE), .groups = 'drop') %>%
  dplyr::mutate(Fuente = "Crudo (sin corrección)")

# 2. Preparar perfil del modelo corregido
perfil_cor <- pred_cor %>%
  dplyr::group_by(Region, month) %>%
  dplyr::summarise(Precipitacion = mean(pred_pc, na.rm = TRUE), .groups = 'drop') %>%
  dplyr::mutate(Fuente = "Corregido (CAS+γ+PC)")

# 3. Preparar perfil observado (usamos cualquiera de los dos dataframes, por ejemplo pred_cor)
perfil_obs <- pred_cor %>%
  dplyr::group_by(Region, month) %>%
  dplyr::summarise(Precipitacion = mean(prcp, na.rm = TRUE), .groups = 'drop') %>%
  dplyr::mutate(Fuente = "Observado")

# 4. Combinar los tres perfiles
perfil_completo <- dplyr::bind_rows(perfil_obs, perfil_crudo, perfil_cor)

# 5. Ordenar los meses (1 a 12, con etiquetas abreviadas)
perfil_completo$month <- factor(perfil_completo$month, levels = 1:12, labels = month.abb)

# 6. Gráfico con tres líneas
p_estacional_completo <- ggplot(perfil_completo, aes(x = month, y = Precipitacion, 
                                                     group = Fuente, color = Fuente)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_wrap(~Region, scales = "free_y") +
  scale_color_manual(values = c("Observado" = "black", 
                                "Crudo (sin corrección)" = "#3B82F6", 
                                "Corregido (CAS+γ+PC)" = "#EF4444")) +
  theme_bw(base_size = 11) +
  labs(
    title = "Observed vs. Raw vs. Corrected Mean Annual Cycle",
    subtitle = "RFSI without corrections vs. RFSI with CAS + γ + PC post-processing",
    x = "Month",
    y = "Mean Daily Precipitation (mm/day)",
    color = "Dataset"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
    plot.subtitle = element_text(size = 9, face = "italic", color = "grey30", hjust = 0.5),
    strip.background = element_rect(fill = "grey95"),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

print(p_estacional_completo)
# ggsave(paste0(dir_salida, "Ciclo_Anual_Observado_Crudo_Corregido.png"), 
#        p_estacional_completo, width = 10, height = 5, dpi = 300)
# 
# cat("\n[+] Gráfico guardado en:", dir_salida, "Ciclo_Anual_Observado_Crudo_Corregido.png\n")


# ==============================================================================
# GRÁFICO MEJORADO DEL CICLO ANUAL: OBSERVADO vs CRUDO vs CORREGIDO
# ==============================================================================
cat("\n[+] Generando gráfico estacional combinado (Observado + Crudo + Corregido)...\n")

# 1. Calcular número de estaciones únicas por región (para añadirlo a las facetas)
n_estaciones_region <- data %>%
  distinct(staid, Region) %>%
  count(Region) %>%
  mutate(Region_label = sprintf("%s (N=%d estaciones)", Region, n))

# 2. Preparar perfiles mensuales
perfil_crudo <- pred_raw %>%
  group_by(Region, month) %>%
  summarise(Precipitacion = mean(pred_rfsi, na.rm = TRUE), .groups = 'drop') %>%
  mutate(Fuente = "Crudo (RFSI sin corrección)")

perfil_cor <- pred_cor %>%
  group_by(Region, month) %>%
  summarise(Precipitacion = mean(pred_pc, na.rm = TRUE), .groups = 'drop') %>%
  mutate(Fuente = "Corregido (CAS+γ+PC)")

perfil_obs <- pred_cor %>%
  group_by(Region, month) %>%
  summarise(Precipitacion = mean(prcp, na.rm = TRUE), .groups = 'drop') %>%
  mutate(Fuente = "Observado (estaciones)")

# 3. Combinar y etiquetar meses
perfil_completo <- bind_rows(perfil_obs, perfil_crudo, perfil_cor) %>%
  mutate(month = factor(month, levels = 1:12, labels = month.abb))

# 4. Unir con las etiquetas de región (para que aparezca el número de estaciones en la faceta)
perfil_completo <- perfil_completo %>%
  left_join(n_estaciones_region, by = "Region") %>%
  mutate(Region_plot = Region_label)  # nueva columna para faceta

# 5. Paleta de colores profesional (accesible para daltónicos)
colores_ciclo <- c("Observado (estaciones)" = "#000000",
                   "Crudo (RFSI sin corrección)" = "#0072B2",   # azul fuerte
                   "Corregido (CAS+γ+PC)" = "#D55E00")          # naranja/terracota

# 6. Gráfico mejorado
p_estacional_completo <- ggplot(perfil_completo,
                                aes(x = month, y = Precipitacion,
                                    group = Fuente, color = Fuente)) +
  geom_line(linewidth = 0.9, alpha = 0.9) +
  geom_point(size = 1.5, alpha = 0.7) +
  facet_wrap(~Region_plot, scales = "free_y", nrow = 1) +
  scale_color_manual(values = colores_ciclo) +
  scale_x_discrete(breaks = month.abb[c(1,4,7,10)],  # etiquetas cada 3 meses
                   labels = month.abb[c(1,4,7,10)]) +
  labs(
    title = "Mean annual cycle of daily precipitation",
    subtitle = "Comparison between observed, raw RFSI, and corrected RFSI (CAS+γ+PC)",
    x = "Month",
    y = "Mean daily precipitation (mm day⁻¹)",
    color = "Dataset"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5, margin = margin(b = 4)),
    plot.subtitle = element_text(size = 9, face = "italic", color = "#2c3e50", hjust = 0.5, margin = margin(b = 10)),
    strip.background = element_rect(fill = "#f5f5f5", color = "#cccccc"),
    strip.text = element_text(face = "bold", size = 10.5, margin = margin(t = 4, b = 4)),
    axis.text.x = element_text(angle = 0, vjust = 0.5, size = 8.5),
    axis.text.y = element_text(size = 8.5),
    axis.title = element_text(face = "bold", size = 10),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 9),
    legend.text = element_text(size = 8.5),
    legend.key.width = unit(1.2, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  ) +
  guides(color = guide_legend(nrow = 1, override.aes = list(linewidth = 1.2)))

# Mostrar gráfico
print(p_estacional_completo)

# Guardar en alta resolución
ggsave(paste0(dir_salida, "Figura_Ciclo_Anual_Observado_Crudo_Corregido.png"),
       p_estacional_completo, width = 10, height = 5, dpi = 350, bg = "white")

cat("[+] Gráfico guardado en:", paste0(dir_salida, "Figura_Ciclo_Anual_Observado_Crudo_Corregido.png\n"))
















# ==============================================================================
# 8. CÁLCULO DE MÉTRICAS GLOBALES POR REGIÓN FISIOGRÁFICA
# ==============================================================================
cat("\n[+] Calculando tabla de métricas zonales (Selva vs. Sierra)... \n")

tabla_zonal <- pred_cor %>%
  dplyr::group_by(Region) %>%
  dplyr::summarise(
    N    = dplyr::n(),
    RMSE = sqrt(mean((prcp - pred_pc)^2, na.rm = TRUE)),
    MAE  = mean(abs(prcp - pred_pc), na.rm = TRUE),
    BIAS = (sum(pred_pc, na.rm = TRUE) - sum(prcp, na.rm = TRUE)) / sum(prcp, na.rm = TRUE) * 100,
    
    # Eficiencia KGE Zonal
    KGE  = {
      obs <- prcp; sim <- pred_pc
      idx <- complete.cases(obs, sim)
      obs <- obs[idx]; sim <- sim[idx]
      if(length(obs) > 5) {
        r_cor <- cor(obs, sim)
        alpha <- sd(sim) / sd(obs)
        beta  <- mean(sim) / mean(obs)
        1 - sqrt((r_cor - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
      } else { NA_real_ }
    },
    
    # Métricas de detección de eventos (Umbral de lluvia: 0.1 mm/dia)
    POD  = sum(pred_pc > 0.1 & prcp > 0.1, na.rm = TRUE) / sum(prcp > 0.1, na.rm = TRUE),
    FAR  = sum(pred_pc > 0.1 & prcp <= 0.1, na.rm = TRUE) / sum(pred_pc > 0.1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(Region)

# Mostrar la tabla en la consola de R
print(tabla_zonal)

# Opcional: Guardar la tabla en un archivo CSV para tu manuscrito
# write.csv(tabla_zonal, paste0(dir_salida, "Metricas_Zonales_Corregidas.csv"), row.names = FALSE)


# ==============================================================================
# MÉTRICAS POR REGIÓN: RFSI CRUDO vs RFSI CORREGIDO (CAS+γ+PC)
# ==============================================================================

library(dplyr)

# Función auxiliar para calcular todas las métricas en un dataframe
# (similar a la que usaste en la validación cruzada, pero adaptada para agrupar por región)
metricas_por_region <- function(df, pred_col, modelo_nombre) {
  df %>%
    group_by(Region) %>%
    summarise(
      N = n(),
      RMSE = sqrt(mean((prcp - !!sym(pred_col))^2, na.rm = TRUE)),
      MAE = mean(abs(prcp - !!sym(pred_col)), na.rm = TRUE),
      R2 = cor(prcp, !!sym(pred_col), use = "complete.obs")^2,
      BIAS = (sum(!!sym(pred_col), na.rm = TRUE) - sum(prcp, na.rm = TRUE)) / sum(prcp, na.rm = TRUE) * 100,
      KGE = {
        obs <- prcp; sim <- !!sym(pred_col)
        idx <- complete.cases(obs, sim)
        obs <- obs[idx]; sim <- sim[idx]
        if (length(obs) > 1) {
          r <- cor(sim, obs)
          alpha <- sd(sim) / sd(obs)
          beta <- mean(sim) / mean(obs)
          1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
        } else { NA_real_ }
      },
      POD = {
        hits <- sum(!!sym(pred_col) >= 0.1 & prcp >= 0.1, na.rm = TRUE)
        miss <- sum(!!sym(pred_col) < 0.1 & prcp >= 0.1, na.rm = TRUE)
        if (hits + miss > 0) hits / (hits + miss) else NA_real_
      },
      FAR = {
        hits <- sum(!!sym(pred_col) >= 0.1 & prcp >= 0.1, na.rm = TRUE)
        fals <- sum(!!sym(pred_col) >= 0.1 & prcp < 0.1, na.rm = TRUE)
        if (hits + fals > 0) fals / (hits + fals) else NA_real_
      },
      CSI = {
        hits <- sum(!!sym(pred_col) >= 0.1 & prcp >= 0.1, na.rm = TRUE)
        fals <- sum(!!sym(pred_col) >= 0.1 & prcp < 0.1, na.rm = TRUE)
        miss <- sum(!!sym(pred_col) < 0.1 & prcp >= 0.1, na.rm = TRUE)
        if (hits + fals + miss > 0) hits / (hits + fals + miss) else NA_real_
      },
      ACC = {
        hits <- sum(!!sym(pred_col) >= 0.1 & prcp >= 0.1, na.rm = TRUE)
        fals <- sum(!!sym(pred_col) >= 0.1 & prcp < 0.1, na.rm = TRUE)
        miss <- sum(!!sym(pred_col) < 0.1 & prcp >= 0.1, na.rm = TRUE)
        cn   <- sum(!!sym(pred_col) < 0.1 & prcp < 0.1, na.rm = TRUE)
        total <- hits + fals + miss + cn
        if (total > 0) (hits + cn) / total else NA_real_
      },
      .groups = "drop"
    ) %>%
    mutate(Modelo = modelo_nombre, .before = 1)
}

# Calcular métricas para el modelo CRUDO (pred_rfsi)
metricas_crudo_region <- metricas_por_region(pred_raw, "pred_rfsi", "Crudo (RFSI sin corrección)")

# Calcular métricas para el modelo CORREGIDO (pred_pc)
metricas_corregido_region <- metricas_por_region(pred_cor, "pred_pc", "Corregido (CAS+γ+PC)")

# Mostrar resultados en consola
cat("\n================ MÉTRICAS POR REGIÓN ================\n\n")
cat("--- MODELO CRUDO ---\n")
print(metricas_crudo_region)
cat("\n--- MODELO CORREGIDO ---\n")
print(metricas_corregido_region)

# Opcional: combinar en una sola tabla para comparación directa
comparativa_region <- bind_rows(metricas_crudo_region, metricas_corregido_region) %>%
  select(Modelo, Region, N, RMSE, MAE, R2, KGE, BIAS, POD, FAR, CSI, ACC)

cat("\n================ TABLA COMPARATIVA ================\n")
print(comparativa_region)

# Guardar resultados en CSV
write.csv(comparativa_region, paste0(dir_salida, "Metricas_por_Region_Crudo_vs_Corregido.csv"), row.names = FALSE)
cat("\nResultados guardados en:", paste0(dir_salida, "Metricas_por_Region_Crudo_vs_Corregido.csv"), "\n")




































# ==============================================================================
# ANÁLISIS DE SENSIBILIDAD DE PARÁMETROS (CAS, γ, PC) - SCRIPT COMPLETO
# ==============================================================================
# Se ejecuta después del script principal (data, folds, fórmula ya definidos)
# Evalúa el impacto en KGE y BIAS sobre el Fold 1 variando cada parámetro.
# ==============================================================================

cat("\n[+] Iniciando análisis de sensibilidad sobre Fold 1...\n")

# 1. Preparar datos de entrenamiento y prueba para el Fold 1
train_cv <- data[data$fold_spatial != 1, ]
test_cv  <- data[data$fold_spatial == 1, ] %>% dplyr::filter(!is.na(prcp))

# 2. Entrenar modelo RFSI base una sola vez (con hiperparámetros fijos)
cat("Entrenando modelo base RFSI...\n")
mod_rfsi <- rfsi(formula = fm.RFSI, data = train_cv,
                 data.staid.x.y.z = c("staid", "x_utm", "y_utm", "z"),
                 n.obs = 5, 
                 cpus = max(1, parallel::detectCores() - 2), 
                 num.trees = 200)

# 3. Obtener predicciones base en entrenamiento y prueba
pred_train <- meteo::pred.rfsi(mod_rfsi, train_cv, "prcp",
                               c("staid","x_utm","y_utm","z"),
                               newdata = train_cv,
                               newdata.staid.x.y.z = c("staid","x_utm","y_utm","z"))
pred_test  <- meteo::pred.rfsi(mod_rfsi, train_cv, "prcp",
                               c("staid","x_utm","y_utm","z"),
                               newdata = test_cv,
                               newdata.staid.x.y.z = c("staid","x_utm","y_utm","z"))

train_cv$pred_rfsi <- pmax(0, pred_train$pred)
test_cv$pred_rfsi  <- pmax(0, pred_test$pred)

# 4. Función que aplica corrección con parámetros variables y retorna KGE y BIAS
evaluar_parametros <- function(lim_inf_CAS = 0.6, lim_sup_CAS = 1.8,
                               lim_inf_gamma = 0.5, lim_sup_gamma = 2.0,
                               factor_cobertura_min = 0.15,
                               umbral_pct = 0.05) {
  
  # --- Umbral mensual p99 ---
  umbral_mensual_train <- train_cv %>%
    dplyr::filter(prcp > 0) %>%
    dplyr::group_by(Region, month) %>%
    dplyr::summarise(p99 = quantile(prcp, 0.99, na.rm = TRUE), .groups = 'drop')
  
  # --- Factores CAS con límites variables ---
  factors_train <- train_cv %>%
    dplyr::group_by(Region, year, month, staid) %>%
    dplyr::summarise(obs_mes = sum(prcp), pred_mes = sum(pred_rfsi), .groups = "drop") %>%
    dplyr::mutate(factor_estacion = ifelse(pred_mes > 0, obs_mes / pred_mes, 1.0)) %>%
    dplyr::group_by(Region, year, month) %>%
    dplyr::summarise(factor_regional = median(factor_estacion, na.rm = TRUE),
                     n_est = n(), .groups = "drop") %>%
    dplyr::mutate(
      factor_regional = ifelse(n_est < 3, 1.0, factor_regional),
      factor_regional = pmax(lim_inf_CAS, pmin(lim_sup_CAS, factor_regional)),
      factor_suavizado = zoo::rollapply(factor_regional, width = 5, FUN = median,
                                        fill = factor_regional, align = "center")
    )
  
  # --- Estadísticas mensuales para γ con límites variables ---
  stats_train <- train_cv %>%
    dplyr::group_by(Region, month) %>%
    dplyr::summarise(
      mean_obs = mean(prcp, na.rm = TRUE),
      sd_obs = sd(prcp, na.rm = TRUE),
      mean_pred = mean(pred_rfsi, na.rm = TRUE),
      sd_pred = sd(pred_rfsi, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      gamma = sd_obs / sd_pred,
      gamma = pmax(lim_inf_gamma, pmin(lim_sup_gamma, gamma))
    )
  
  # --- Estadísticas diarias para PC ---
  total_estaciones <- n_distinct(train_cv$staid)
  umbral_dinamico <- max(2, round(umbral_pct * total_estaciones))
  
  estadisticas_diarias <- train_cv %>%
    dplyr::group_by(Region, date) %>%
    dplyr::summarise(
      estaciones_con_lluvia = sum(prcp > 0.1, na.rm = TRUE),
      total_estaciones = n(),
      umbral_inferior = max(0.5, quantile(prcp, 0.15, na.rm = TRUE)),
      .groups = "drop"
    )
  
  # --- Aplicar correcciones al conjunto de prueba ---
  test_cv_cor <- test_cv %>%
    dplyr::left_join(factors_train %>% dplyr::select(Region, year, month, factor_suavizado),
                     by = c("Region", "year", "month")) %>%
    dplyr::left_join(stats_train %>% dplyr::select(Region, month, mean_obs, mean_pred, gamma),
                     by = c("Region", "month")) %>%
    dplyr::left_join(estadisticas_diarias, by = c("Region", "date")) %>%
    dplyr::left_join(umbral_mensual_train, by = c("Region", "month")) %>%
    dplyr::mutate(
      factor_suavizado = ifelse(is.na(factor_suavizado), 1.0, factor_suavizado),
      pred_cas = pred_rfsi * factor_suavizado,
      mean_obs = ifelse(is.na(mean_obs), 0, mean_obs),
      mean_pred = ifelse(is.na(mean_pred), 0, mean_pred),
      gamma = ifelse(is.na(gamma), 1.0, gamma),
      pred_var = mean_obs + gamma * (pred_cas - mean_pred),
      pred_var = pmax(0, pred_var),
      p99 = ifelse(is.na(p99), Inf, p99),
      # Factor de cobertura suave con mínimo variable
      factor_cobertura = pmin(1.0, estaciones_con_lluvia / umbral_dinamico),
      factor_cobertura = pmax(factor_cobertura_min, factor_cobertura),
      pred_pc = pred_var * factor_cobertura,
      # Reglas adicionales
      pred_pc = ifelse(estaciones_con_lluvia >= umbral_dinamico & pred_pc < umbral_inferior, 0, pred_pc),
      pred_pc = ifelse(pred_pc > p99, p99, pred_pc),
      pred_pc = ifelse(total_estaciones > 0 & estaciones_con_lluvia == 0, 0, pred_pc),
      pred_pc = pmax(0, pred_pc)
    )
  
  # Calcular KGE y BIAS
  obs <- test_cv_cor$prcp
  sim <- test_cv_cor$pred_pc
  idx <- complete.cases(obs, sim)
  obs <- obs[idx]
  sim <- sim[idx]
  if (length(obs) == 0) return(data.frame(KGE = NA, BIAS = NA))
  
  r <- cor(sim, obs)
  alpha <- sd(sim) / sd(obs)
  beta <- mean(sim) / mean(obs)
  kge <- 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
  bias <- (sum(sim) - sum(obs)) / sum(obs) * 100
  
  return(data.frame(KGE = kge, BIAS = bias))
}

# ----------------------------------------------------------------------------
# 5. DEFINIR LOS VALORES A PROBAR PARA CADA PARÁMETRO
# ----------------------------------------------------------------------------
valores_CAS_lim_inf  <- c(0.4, 0.5, 0.6, 0.7, 0.8)
valores_CAS_lim_sup  <- c(1.5, 1.6, 1.8, 2.0, 2.2)
valores_gamma_lim_inf <- c(0.3, 0.4, 0.5, 0.6, 0.7)
valores_gamma_lim_sup <- c(1.5, 1.8, 2.0, 2.2, 2.5)
valores_cob_min       <- c(0.05, 0.10, 0.15, 0.20, 0.30)
valores_umbral_pct    <- c(0.02, 0.03, 0.05, 0.07, 0.10)

# Almacenar resultados
resultados <- list()

# ---- Sensibilidad al límite inferior de CAS ----
cat("Evaluando: límite inferior de CAS\n")
for (v in valores_CAS_lim_inf) {
  res <- evaluar_parametros(lim_inf_CAS = v, lim_sup_CAS = 1.8,
                            lim_inf_gamma = 0.5, lim_sup_gamma = 2.0,
                            factor_cobertura_min = 0.15, umbral_pct = 0.05)
  resultados$CAS_inf <- rbind(resultados$CAS_inf, data.frame(Parametro = "CAS_inf", Valor = v, res))
}

# ---- Sensibilidad al límite superior de CAS ----
cat("Evaluando: límite superior de CAS\n")
for (v in valores_CAS_lim_sup) {
  res <- evaluar_parametros(lim_inf_CAS = 0.6, lim_sup_CAS = v,
                            lim_inf_gamma = 0.5, lim_sup_gamma = 2.0,
                            factor_cobertura_min = 0.15, umbral_pct = 0.05)
  resultados$CAS_sup <- rbind(resultados$CAS_sup, data.frame(Parametro = "CAS_sup", Valor = v, res))
}

# ---- Sensibilidad al límite inferior de gamma ----
cat("Evaluando: límite inferior de gamma\n")
for (v in valores_gamma_lim_inf) {
  res <- evaluar_parametros(lim_inf_CAS = 0.6, lim_sup_CAS = 1.8,
                            lim_inf_gamma = v, lim_sup_gamma = 2.0,
                            factor_cobertura_min = 0.15, umbral_pct = 0.05)
  resultados$gamma_inf <- rbind(resultados$gamma_inf, data.frame(Parametro = "gamma_inf", Valor = v, res))
}

# ---- Sensibilidad al límite superior de gamma ----
cat("Evaluando: límite superior de gamma\n")
for (v in valores_gamma_lim_sup) {
  res <- evaluar_parametros(lim_inf_CAS = 0.6, lim_sup_CAS = 1.8,
                            lim_inf_gamma = 0.5, lim_sup_gamma = v,
                            factor_cobertura_min = 0.15, umbral_pct = 0.05)
  resultados$gamma_sup <- rbind(resultados$gamma_sup, data.frame(Parametro = "gamma_sup", Valor = v, res))
}

# ---- Sensibilidad al factor de cobertura mínimo ----
cat("Evaluando: factor de cobertura mínimo\n")
for (v in valores_cob_min) {
  res <- evaluar_parametros(lim_inf_CAS = 0.6, lim_sup_CAS = 1.8,
                            lim_inf_gamma = 0.5, lim_sup_gamma = 2.0,
                            factor_cobertura_min = v, umbral_pct = 0.05)
  resultados$cob_min <- rbind(resultados$cob_min, data.frame(Parametro = "cob_min", Valor = v, res))
}

# ---- Sensibilidad al porcentaje para umbral dinámico ----
cat("Evaluando: porcentaje umbral dinámico\n")
for (v in valores_umbral_pct) {
  res <- evaluar_parametros(lim_inf_CAS = 0.6, lim_sup_CAS = 1.8,
                            lim_inf_gamma = 0.5, lim_sup_gamma = 2.0,
                            factor_cobertura_min = 0.15, umbral_pct = v)
  resultados$umbral_pct <- rbind(resultados$umbral_pct, data.frame(Parametro = "umbral_pct", Valor = v, res))
}

# ----------------------------------------------------------------------------
# 6. COMBINAR Y MOSTRAR RESULTADOS
# ----------------------------------------------------------------------------
library(dplyr)
tabla_sensibilidad <- bind_rows(resultados)
cat("\n\n================ TABLA DE SENSIBILIDAD ================\n")
print(tabla_sensibilidad)

# Guardar en CSV
write.csv(tabla_sensibilidad, paste0(dir_salida, "Sensibilidad_parametros.csv"), row.names = FALSE)

# ----------------------------------------------------------------------------
# 7. GRÁFICOS DE SENSIBILIDAD
# ----------------------------------------------------------------------------
library(ggplot2)
p_sens <- ggplot(tabla_sensibilidad, aes(x = Valor, y = KGE)) +
  geom_line(linewidth = 1) + 
  geom_point(size = 2) +
  facet_wrap(~Parametro, scales = "free_x") +
  labs(title = "Sensitivity Analysis of Correction Parameters (Fold 1)",
       y = "KGE", x = "Parameter value") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))
print(p_sens)
ggsave(paste0(dir_salida, "Sensibilidad_KGE.png"), p_sens, width = 10, height = 6, dpi = 300)

cat("\n[+] Análisis de sensibilidad completado. Resultados guardados en:", dir_salida, "\n")



