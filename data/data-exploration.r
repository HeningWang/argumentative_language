library(tidyverse)
library(brms)
library(aida)
library(cspplot)

# these options help Stan run faster
options(mc.cores = parallel::detectCores(),
        brms.backend = "cmdstanr")

# global color scheme from CSP
project_colors = cspplot::list_colors() |> pull(hex)

# setting theme colors globally
scale_colour_discrete <- function(...) {
  scale_colour_manual(..., values = project_colors)
}
scale_fill_discrete <- function(...) {
  scale_fill_manual(..., values = project_colors)
}

d_pilot <- read_csv("data_pilot/cleaned_data.csv") |> 
  mutate(source = "pilot")
d_exp1and2 <- read_csv("data_experiment2/cleaned_data_1and2.csv") 

# number of participants in the pilot study
d_pilot |> nrow() / 20

d_exps <- d_exp1and2 |> 
  mutate(arrayCondition = case_when(
    array_size_condition == "wideShort" ~ " 5 x 12",
    array_size_condition == "narrowShort" ~ " 5 x 6",
    array_size_condition == "wideLong" ~ "11 x 12",
    array_size_condition == "narrowLong" ~ "11 x  6"
  )) |> 
  mutate(condition = ifelse(condition, "high", "low") |> factor(levels = c("high", "low")))


# total number of participants after cleaning
n_participants <- d_exps |> pull(id) |> unique() |> length()
# total number of additionally excluded responses (due to semantic falsity) after cleaning
n_participants * 20 - nrow(d_exps)

# number of participants assigned to each size condition
d_exps |> count(arrayCondition) |>  mutate(n = n / 20)

# prettify the response strings

responses_prettified <- d_exps |> pull(response) |> 
  # 1) drop [, ], and '
  str_remove_all("\\[|\\]|'") %>%     
  # 2) collapse commas into " | "
  str_replace_all(",\\s*", " | ") %>% 
  # 3) trim any stray whitespace
  str_trim()

d_exps$response <- responses_prettified


## sum stats & basic plot for aggregate data ----

sum_stats <- d_exps |> 
  group_by(condition, arrayCondition) |> count(response) |> 
  mutate(proportion = n / sum(n))

sum_stats |> 
  ggplot(aes(x = response , y = proportion, group = condition, fill = condition)) +
  geom_bar(stat = "identity", position = "dodge") + 
  facet_grid(arrayCondition ~ ., scale = "free") +
  theme_csp() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("pics/barplot_responses_perConditions.pdf", width = 8, height = 4.5, scale = 1.5)

############# TODO revision up to HERE ##############


## sum stats & basic plot for selected situations data ----

sum_stats_full <- d_exps |> 
  group_by(condition, arrayCondition, row_number) |> count(response) |> 
  mutate(proportion = n / sum(n)) 

sum_stats_full |> 
  
  filter(row_number == "[12, 12, 9, 3, 3]") |> 
  ggplot(aes(x = response , y = proportion, group = condition, fill = condition)) +
  geom_bar(stat = "identity", position = "dodge") + 
  theme_csp() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("pics/barplot_responses_example12|12|9|3|3.pdf", width = 8, height = 4.5, scale = 1)

## regression models ----


# fit_noCondition <- brms::brm(
#   formula = response ~ arrayAndSize,
#   data = d_exps |> mutate(arrayAndSize = paste(arrayCondition, studentsArray)),
#   family = categorical()
# )
# 
# fit_full <- brms::brm(
#   formula = response ~ arrayAndSize * condition,
#   data = d_exps |> mutate(arrayAndSize = paste(arrayCondition, studentsArray)),
#   family = categorical()
# )



