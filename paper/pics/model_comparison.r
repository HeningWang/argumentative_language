library(tidyverse)
library(aida)
library(cspplot)

# global color scheme from CSP
project_colors = cspplot::list_colors() |> pull(hex)

# setting theme colors globally
scale_colour_discrete <- function(...) {
  scale_colour_manual(..., values = project_colors)
}
scale_fill_discrete <- function(...) {
  scale_fill_manual(..., values = project_colors)
}


# z test comparing first and second best model
zTest <- function(loo, se) {
  1 - pnorm(loo, se)
}
loo_diff_combined <- 32.962128
se_diff_combined  <- 31.663264
zTest(loo_diff_combined, se_diff_combined)
# => not a significant difference between the two models




d <- read_csv("model_comparison-combined.csv") |> 
  # cast 'model' as an ordered factor with ordering from colums 'loo'
  mutate(model = fct_reorder(factor(model), loo)) |> 
  mutate(analysis = factor(analysis, level = c("population", "hierarchical")))

d |> ggplot(aes(x = model, y = loo, group = analysis, color = analysis)) +
  geom_errorbar(aes(ymin = loo - se, ymax = loo + se),
              position = position_dodge(width = 0.5), width = 0.1) +
  # dodge points by group; geometric shape by factor 'analysis'
  geom_point(aes(shape = analysis), position = position_dodge(width = 0.5), size = 3) +
  coord_flip() +
  labs(x = NULL, y = NULL) +
  theme_aida() +
  # show legend on at the top
  theme(legend.position = "top",
        # legend.title = element_blank(),
        legend.text = element_text(size = 10),
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10),
        plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

ggsave("model_comparison-combined.pdf", width = 5, height = 2.5, scale = 0.9)
ggsave("../../paper_CogSci-2026/pics/model_comparison-combined.pdf", width = 5, height = 2.5, scale = 0.9)
