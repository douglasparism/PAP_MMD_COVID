
# Librerias + Datos -------------------------------------------------------

#Load packages
library("tidyverse")
library("beepr")
library("tictoc")
library("tibbletime")

#Modelado
library(tidymodels)
library(themis)
library(ranger)
library(plyr)

tic()
datosSinave<- read.csv('C:/Users/DParis/OneDrive/ITESO/PAP MMD/S2/Rt/Datos/210701COVID19MEXICO.csv')
beep(4)
toc()



# Tratado de datos completo -----------------------------------------------

datosSinave2_2 <- datosSinave %>% filter(CLASIFICACION_FINAL <= 3, # Positivos
                                         as.Date(FECHA_INGRESO) >= as.Date('2020-06-01')) %>% 
  mutate(DIAS_HOSPITALIZACION = as.integer(as.Date(FECHA_INGRESO)-as.Date(FECHA_SINTOMAS)),
         DEFUNCIONES = if_else(FECHA_DEF == "9999-99-99",0,1),
         INTUBADO = if_else(INTUBADO==1,"yes","not"),
         NEUMONIA = if_else(NEUMONIA==1,1,0),
         SEXO = if_else(SEXO==1,1,0),
         TIPO_PACIENTE = if_else(TIPO_PACIENTE==1,1,0),
         DIABETES = if_else(DIABETES==1,1,0),
         EPOC = if_else(EPOC==1,1,0),
         ASMA = if_else(ASMA==1,1,0),
         INMUSUPR = if_else(INMUSUPR==1,1,0),
         HIPERTENSION = if_else(HIPERTENSION==1,1,0),
         INDIGENA = if_else(INDIGENA==1,1,0),
         CARDIOVASCULAR = if_else(CARDIOVASCULAR==1,1,0),
         OBESIDAD = if_else(OBESIDAD==1,1,0),
         RENAL_CRONICA = if_else(RENAL_CRONICA==1,1,0),
         TABAQUISMO = if_else(TABAQUISMO==1,1,0),
         NEUMONIA = if_else(NEUMONIA==1,1,0),
         UCI = if_else(UCI==1,1,0),
         OTRA_COM = if_else(OTRA_COM==1,1,0)) %>% 
  filter(ENTIDAD_RES < 36) %>%
  select(-FECHA_ACTUALIZACION,-EMBARAZO,-ENTIDAD_UM,-ORIGEN,
         -MIGRANTE,-PAIS_ORIGEN,-PAIS_NACIONALIDAD,-CLASIFICACION_FINAL,
         -RESULTADO_LAB,-TOMA_MUESTRA_LAB,-TOMA_MUESTRA_ANTIGENO,-RESULTADO_ANTIGENO,
         -HABLA_LENGUA_INDIG,-NACIONALIDAD,-FECHA_SINTOMAS,-FECHA_INGRESO,
         -SECTOR,-ENTIDAD_NAC,
         -FECHA_DEF, -UCI,-OTRO_CASO) %>%    
  mutate_if(is.character, factor)
beep(1)


# Muestra Aleatoria -------------------------------------------------------


n_sample <- 10000
datosSinave1_1 <- sample_n(datosSinave2_2, size= n_sample) %>%
  select(-MUNICIPIO_RES)# tomar una muestra aleatoria

datosSinave1_1$ENTIDAD_RES <- as.factor(datosSinave1_1$ENTIDAD_RES)


####### Modelo Bootstrp (datos simulados)

sinave_boot <- bootstraps(datosSinave1_1) 
beep(1)

# Preparación de los datos 

sinave_rec <- recipe(INTUBADO ~ ., # Variable objetivo
                     data = datosSinave1_1) %>% #Datos de entrenamiento
  update_role(ID_REGISTRO, new_role = "Id") %>% 
  step_naomit(all_predictors()) %>%
  step_other(ENTIDAD_RES) %>%
  step_dummy(ENTIDAD_RES) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>% 
  step_smote(INTUBADO)
beep(1)

sinave_prep <- prep(sinave_rec)
beep(1)

J <- juice(sinave_prep)


# Modelo (tuning) ------------------------------------------------------------------

# Definicion del modelo
rf_spec <- rand_forest(trees = 1000, mtry = tune(), min_n = tune()) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# Workflow
sinave_wf <- workflow() %>%
  add_recipe(sinave_rec) %>%
  add_model(rf_spec)

sinave_folds <- vfold_cv(datosSinave1_1)

tune_res <- tune_grid(
  sinave_wf,
  resamples = sinave_folds,
  grid = 20
)

tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

rf_grid <- grid_regular(
  mtry(range = c(0, 10)),
  min_n(range = c(0, 40)),
  levels = 5
)

regular_res <- tune_grid(
  sinave_wf,
  resamples = sinave_folds,
  grid = rf_grid
)

regular_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")


best_auc <- select_best(regular_res, "roc_auc")

final_rf <- finalize_model(
  rf_spec,
  best_auc
)

beep(1)

# Modelo (ejec) -----------------------------------------------------------

# Definicion del modelo
rf_spec <- rand_forest(trees = 1000, mtry = 5, min_n = 10) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# Workflow
sinave_wf <- workflow() %>%
  add_recipe(sinave_rec) %>%
  add_model(rf_spec)

tic()
sinave_res <- fit_resamples(
  sinave_wf,
  resamples = sinave_boot,
  control = control_resamples(save_pred = TRUE)
)
toc()
beep(1)


#Matriz confusión
m <- sinave_res %>%
  collect_predictions() %>%
  conf_mat(INTUBADO, .pred_class)


accuracy<-summary(m)%>% slice(1L)
accuracy<-accuracy[3]$.estimate
ppv_ac<-summary(m)%>% slice(5L)
ppv_ac<-ppv_ac[3]$.estimate
npv_ac<-summary(m)%>% slice(6L)
npv_ac<-npv_ac[3]$.estimate


# Var Importance ----------------------------------------------------------


library(vip)

tic()
importance <- rf_spec %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(
    INTUBADO ~ .,
    data = juice(sinave_prep) %>%
      select(-ID_REGISTRO) 
  ) 
toc()
beep(1)

importance %>%
  vip(geom = "col")

