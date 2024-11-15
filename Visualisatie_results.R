## Startup #####################################################################

options(scipen = 999)

if (!require("readxl")) install.packages("readxl");library(readxl)
if (!require("lubridate")) install.packages("lubridate");library(lubridate)
if (!require("tidyverse")) install.packages("tidyverse");library(tidyverse)
if (!require("grid")) install.packages("grid");library(grid)
if (!require("gridExtra")) install.packages("gridExtra");library(gridExtra)
if (!require("ggpubr")) install.packages("ggpubr");library(ggpubr)
if (!require("vctrs")) install.packages("vctrs");library(vctrs)

## Gneral functions ############################################################

plot_function_color <- function(x,color_var){
  
  ggplot(x, aes(runtime, rmse, label = exp_naam, color = .data[[color_var]])) +
    geom_point()+
    geom_text_repel(
      max.overlaps = 20,           # Increase the max overlaps limit
      size = 3,                    # Adjust label size for readability
      segment.size = 0.2,          # Control line thickness for better visibility
      box.padding = 0.3,           # Padding around the text box
      point.padding = 0.3          # Padding around the point
    ) +
    facet_grid(. ~ use_rear_lidar)+
    theme_bw()+
    labs(title = "Results SLAM algorithm experiments",
         subtitle = paste0("Scatterplot with RMSE versus Runtime, with and without rear lidar sensor. \nColor based on ", 
                           color_var, "-value"),
         color = color_var,
         x = "Runtime [seconds]",
         y = "RMSE [mtr]") +
    theme(legend.position = "right")
  
}

plot_function_boxplot <- function(x, y_value, color_var){
  
  ggplot(x, aes(exp_naam, y_value = .data[[y_value]], fill = .data[[color_var]])) +
    geom_boxplot()+
    facet_grid(. ~ use_rear_lidar)+
    theme_bw()+
    labs(title = "Results SLAM algorithm experiments",
         subtitle = paste0("Boxplot of ",y_value, " of 10 runs, with and without rear lidar sensor,\nColor based on ", 
                           color_var, "-value"),
         x = "Experiment number",
         y = y_value) +
    theme(legend.position = "off",
          axis.text.x = element_text(angle = 90, vjust = 0, hjust=0))
  
}

## Import data #################################################################

map <- "C:\\Users\\michel.marien_icarew\\Documents\\GitHub\\Particle-Filter-SLAM\\experiments\\v5"
bestand <- "\\results.csv"

import_file_results <- import_list(paste0(map, bestand,sep=""),
                           which = c("results"),
                           rbind=T,
                           rbindlabel = "Tabblad",
                           rbind_fill = T)

bestand <- "\\results_mean.csv"

import_file_results_mean <- import_list(paste0(map, bestand,sep=""),
                                   which = c("results"),
                                   rbind=T,
                                   rbindlabel = "Tabblad",
                                   rbind_fill = T)

## Pre procssing mean values ###################################################

df_results_mean <- import_file_results_mean[-1,]

df_results_mean$exp_num <- as.numeric(str_extract(df_results_mean$experiment_file, "[:digit:]+"))
df_results_mean$exp_num <- ifelse(df_results_mean$exp_num>18, df_results_mean$exp_num-18, df_results_mean$exp_num)
df_results_mean$exp_num <- ifelse(df_results_mean$exp_num<10,paste("0",df_results_mean$exp_num, sep=""), df_results_mean$exp_num)

df_results_mean$exp_naam <- paste(str_sub(df_results_mean$experiment_file, 1,3), 
                                  df_results_mean$exp_num,sep="_")

df_results_mean <- df_results_mean %>% select(!exp_num)

df_results_mean$rmse <- as.numeric(sub(",", ".", df_results_mean$rmse, fixed = TRUE))
df_results_mean$runtime <- as.numeric(sub(",", ".", df_results_mean$runtime, fixed = TRUE))

df_results_mean[,3] <- as.character(df_results_mean[,3])
df_results_mean[,4] <- as.character(df_results_mean[,4])
df_results_mean[,5] <- as.character(df_results_mean[,5])
df_results_mean[,6] <- as.character(df_results_mean[,6])
df_results_mean[,7] <- as.character(df_results_mean[,7])

df_results_mean$use_rear_lidar <- ifelse(df_results_mean$use_rear_lidar==0, "Rear sensor Off", "Rear sensor On")

## Create txt for upload Latex #################################################

num_exp = (nrow(df_results_mean)/2)

vc_num_exp <-c(1:num_exp)
vc_named_num_exp <- paste0("Scen_x",vc_num_exp)

vc_scenarios <- rep(c("a", "b"), each=nrow(df_results_mean)/2)

df_latex <- df_results_mean[,c(8,9)]

df_latex$exp <- vec_rep(vc_named_num_exp,2)
df_latex$scn <- vc_scenarios

df_latex <- df_latex %>% mutate(across(c('rmse', 'runtime'), round, 3))

df_latex <- df_latex%>% pivot_wider(names_from = scn, values_from = c(rmse, runtime))
df_latex$table_sep <- "\\\\"

df_latex <- unite(df_latex, 
                  col='Latex', 
                  c("exp",
                    "rmse_a", 
                    "runtime_a", 
                    "rmse_b", 
                    "runtime_b"),
                    sep = " & ")

df_latex <- unite(df_latex, 
                  col='Latex', 
                  c("Latex",
                    "table_sep"),
                  sep = "")

write.csv(df_latex, paste0(map, "\\resultaten_mean_tabel.txt",sep=""), row.names = F)

## Pre procssing individual experiment values ##################################

df_results <- import_file_results[,-2] %>% 
  filter(!experiment_file=="experiments\\experiment00")

df_results$exp_num <- as.numeric(str_extract(df_results$experiment_file, "[:digit:]+"))
df_results$exp_num <- ifelse(df_results$exp_num>18, df_results$exp_num-18, df_results$exp_num)
df_results$exp_num <- ifelse(df_results$exp_num<10,paste("0",df_results$exp_num, sep=""), df_results$exp_num)

df_results$exp_naam <- paste(str_sub(df_results$experiment_file, 1,3), 
                                  df_results$exp_num,sep="_")

df_results <- df_results %>% select(!exp_num)


df_results$rmse <- as.numeric(sub(",", ".", df_results$rmse, fixed = TRUE))
df_results$runtime <- as.numeric(sub(",", ".", df_results$runtime, fixed = TRUE))

df_results[,3] <- as.character(df_results[,3])
df_results[,4] <- as.character(df_results[,4])
df_results[,5] <- as.character(df_results[,5])
df_results[,6] <- as.character(df_results[,6])
df_results[,7] <- as.character(df_results[,7])

df_results$use_rear_lidar <- ifelse(df_results$use_rear_lidar=="FALSE", "Rear sensor Off", "Rear sensor On")

## Plotten van data ############################################################
# Plotten van scatterplots
ns_plot <- plot_function_color(df_results_mean, "noise_sigma")
pc_plot <- plot_function_color(df_results_mean, "particle_count")
lr_plot <- plot_function_color(df_results_mean, "local_search_resolution")

master_title <- textGrob("Visualization of experiment results", gp = gpar(fontsize = 20))

grid.arrange(pc_plot,ns_plot,lr_plot, ncol=1)


plot_overview <- arrangeGrob(ns_plot,pc_plot, lr_plot, lo_plot, ncol= 1 ) 

ggsave("Overview_results_experiments.jpg", plot_overview)

# Plotten van boxplots

box_rmse <- plot_function_boxplot(df_results, "rmse", "use_rear_lidar")
box_run <- plot_function_boxplot(df_results, "runtime", "use_rear_lidar")

grid.arrange(box_rmse,box_run, ncol=1)
