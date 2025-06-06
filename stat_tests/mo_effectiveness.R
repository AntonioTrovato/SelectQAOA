# Load necessary libraries
library(FSA)
library(PMCMRplus)
library(coin)
library(jsonlite)
library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))
getwd() 

# Load JSON data
json_data <- fromJSON("../multi_obj_frontiers_eval.json")

# Helper to map JSON config names to R config names
config_map <- list(
  statevector = "statevector_sim",
  aer = "aer_sim",
  fake_vigo = "fake_vigo",
  fake_vigo_zne = "fake_vigo_zne",
  noise_1 = "depolarizing_sim/01",
  noise_2 = "depolarizing_sim/02",
  noise_5 = "depolarizing_sim/05"
)

programs <- c("flex", "grep", "gzip", "sed")
methods <- c("qtcs" = "selectqa", "div_ga" = "divga", "qaoa" = "qaoa", "add_greedy" = "add_greedy")

configs <- list()

for (config_name in names(config_map)) {
  json_config_key <- config_map[[config_name]]
  configs[[config_name]] <- list()
  
  for (prog in programs) {
    configs[[config_name]][[prog]] <- list()
    
    for (method in names(methods)) {
      json_key <- methods[[method]]
      raw_data <- json_data[[prog]][[json_config_key]][[json_key]]
      
      if (method == "add_greedy") {
        value_list <- rep(raw_data[1], 10)
      } else {
        value_list <- raw_data[[1]]
      }
      
      configs[[config_name]][[prog]][[method]] <- value_list
    }
  }
}

#print(configs)

# Function to compute A12 effect size
A12_effect_size <- function(x, y) {
  n_x <- length(x)
  n_y <- length(y)
  count <- sum(outer(x, y, ">")) + 0.5 * sum(outer(x, y, "=="))
  return(count / (n_x * n_y))
}

# Iterate through each configuration
for (config_name in names(configs)) {
  cat("\n=============================\n")
  cat("CONFIGURATION:", config_name, "\n")
  cat("=============================\n")
  
  # Iterate through each measurement group in the configuration
  for (group_name in names(configs[[config_name]])) {
    cat("\n--- Measurement Group:", group_name, "---\n")
    
    # Extract the data for the current group
    current_data <- configs[[config_name]][[group_name]]
    
    # Convert to dataframe for analysis
    data <- data.frame(
      value = unlist(current_data),
      group = rep(names(current_data), each = 10) # Each group has 10 values
    )
    
    # Perform pairwise Kolmogorov-Smirnov test
    group_list <- split(data$value, data$group)
    ks_results <- data.frame()
    
    for (i in 1:(length(group_list) - 1)) {
      for (j in (i + 1):length(group_list)) {
        ks_test <- ks.test(group_list[[i]], group_list[[j]])
        ks_results <- rbind(ks_results, 
                            data.frame(Group1 = names(group_list)[i], 
                                       Group2 = names(group_list)[j], 
                                       Statistic = ks_test$statistic, 
                                       P_Value = ks_test$p.value))
      }
    }
    
    cat("\nKolmogorov-Smirnov Test Results:\n")
    print(ks_results)
    
    # Perform Kruskal-Wallis test
    kruskal_test <- kruskal.test(value ~ group, data = data)
    print(kruskal_test)
    
    # Perform Dunn's test
    dunn_test <- dunnTest(value ~ group, data = data, method = "bh")
    cat("Dunn's Test (Benjamini-Hochberg correction):\n")
    print(dunn_test$res)  # Print only p-values
    
    # Compute A12 for each pair of groups
    group_list <- lapply(current_data, as.numeric)
    group_names <- names(group_list)
    
    a12_results <- data.frame()
    for (i in 1:(length(group_list) - 1)) {
      for (j in (i + 1):length(group_list)) {
        a12_value <- A12_effect_size(group_list[[i]], group_list[[j]])
        a12_results <- rbind(a12_results, data.frame(Group1 = group_names[i], Group2 = group_names[j], A12 = a12_value))
      }
    }
    
    # Print A12 effect size results
    cat("\nA12 Effect Size (probability that x > y)\n")
    print(a12_results)
  }
}

