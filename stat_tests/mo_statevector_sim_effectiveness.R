# Load necessary libraries
library(FSA)
library(PMCMRplus)
library(coin)
library(jsonlite)

# Carica i dati dal file JSON
json_data <- fromJSON("../statevector_multi_obj_frontiers_eval.json")

# Ristruttura i dati nel formato desiderato
configs <- list(
  statevector = lapply(json_data, function(program_data) {
    # Rinomina le chiavi da "statevector_sim_1" a "rep1", ecc.
    names(program_data) <- sub("statevector_sim_", "rep", names(program_data))
    return(program_data)
  })
)

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

