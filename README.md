## nba_classification_project
#### Welcome to this initial exploration of NBA Team Data from the 1979-80 season until the current 2022-23 season!  The primary purpose of this project is to make accurate mid-season predictions about which teams will make the playoffs at the end of the season. The project data-set was personally curated coutesy of https://www.basketball-reference.com.  Season-wide Team data for each NBA Team from 1979 until present was aggregated and each row was labeled as a "Playoff" team or a "Non-Playoff" team.
#### The goals of this initial exploration are as follows:
- PRIMARY: Generate actionable predictions regarding whether or not a team will qualify for the Playoffs at season's-end.
- SECONDARY: Draw reasonable inferences regarding the characteristics of those teams which qualify for the playoffs, and those that do not.
- SECONDARY: Identify those features which retain the greatest importance for determining the potential playoff status of a particular team.

#### PROJECT DESCRIPTION and ASSUMPTIONS:
- Basketball has changed significantly over the past several generations, with respect to tactics, strategy, and tempo.  Thus, the team statistics generated per game have changed over time to reflect this assertion. 

- Consequently, this dataset does not display team stats on a "per game" basis.  Instead, team stats are displayed "per 100 possessions".  This allows for a somewhat-more-valid comparison to be made between teams of differing generations.  This also partially controls for stats being functionally-dependent upon common in-game conditions and provides a more distilled view of a team's capabilities.

- The 1979-80 NBA Season was chosen as the starting point for the dataset because it represesnts the most recent, fundamental change to the game: the 3-pt Shot.  While, it took over 20 years for the 3-pt shot to actually change the game, it is important to note that the efficacy of the model is assumed to be correlated with the number of datapoints (approx. 1200), and thus, 1979 was chosen over 2003, as this would have only yielded about 600 datapoints.

#### Project Planning:
- Plan: Questions and Hypotheses
- Acquire: Curated dataset from https://www.basketball-reference.com by subsetting team-level statistics which include stats reflecting "skill, talent, coaching, and hustle".  For example, team Field Goal % is a function of skill level(among other contributors), while Steals, Rebounds, or Fouls Drawn are a function of "talent and hustle" (among others).
- Prepare: Kept outliers after investigating their nature, missingness was a non-issue, as there were ZERO entries containing NULL values for predictors.  StandardScaler used for scaling purposes.  Split into ML subsets (Train/Validate/Test).
- Explore: Univariate and multi-variate analysis, correlation matrix, 2D visualization, correlation significance testing, 2-sample T-testing for significant differences in means.
- Model: Established a baseline "Precision" for Positive class of 57.1% using the most frequent target occurance of "yes: playoffs".  Then with a DecisionTreeClassifier with MaxDepth set to 4, established a new Precision floor of 86.0%. After creating models with different tree-based and non-tree-based algorithms and multiple tunings, findings indicated a Multi-Layer Perceptron with a three hidden layers (256,128,64 nodes) yielded best validation results (90.0% Precision on Test).
- Deliver: Please refer to this doc as well as the Final_NBA.ipynb file for the finished version of the presentation, in addition to each of the underlying exploratory notebooks.

#### Initial hypotheses and questions:
* What meaningful features can be leveraged to create a model that predicts whether or not an NBA team will qualify for the playoffs?  
* Can statistics which reflect a team's level of skill, preparation and effort (hustle) yield similar or even better results than the traditional statistics upon which most players/teams are evaluated?  
* Can these chosen features transcend the changing nature of the game in order to allow for an "apples-to-apples" categorization of teams from multiple eras of NBA play? 
* If all of the above are realistic and possible, can an informed sports-wager be placed utilizing the outputs of said modeling?
* Which is the more disasterous outcome, a false positive: a team predicted to make the playoffs but fails to do so, or a false-negative: a team that is predicted to NOT make the playoffs but ultimately does?
* Can the sports-wagering public make use of the model output when placing "futures" wagers on teams that may or may not win NBA Championship, Conf Championship, or Div Championship?

#### Data Dictionary: 

|Feature |  Data type | Definition |
|---|---|---|
| fg_pct: | float | percentage of shots taken which scored |
| opp_fg_pct: | float | same as above, but for the *opponent* |
| three_pt_pct: | float | percentage of 3-point shots taken which scored |
| opp_three_pt_pct: | float | same as above, but for the *opponent* |
| ft_pct: | float | percentage of free throws taken which scored |
| rebounds: | float | number of missed shots recovered from rim/backboard |
| opp_rebounds: | float | same as above, but for the *opponent* |
| assists: | float | number of passes completed to player in the act of scoring a field goal |
| steals: | float | number of times the ball was successfully stripped from opposing player |
| opp_steals: | float | same as above, but for the *opponent* |
| trnovers_committed: | float | number of times the ball was turned-over to the opponent |
| pts: | float | number of points scored |
| opp_pts: | float | same as above, but for the *opponent* |
| opp_fouls: | float | number of times the opponent fouled a player |
| prev-season: | int | 1-made Playoffs in Previous Season / 0-did not make Playoffs in Previous Season |
| playoffs: | int | TARGET: 1-Playoff Team / 0-Non-Playoff Team |


#### Findings, Recommendations, and Takeaways:

- Modeling was optimized for PRECISION for the Positive Class ("Playoffs").  The nature of sports wagering allows False Negatives to be inconsequential, however, False Positives are punished significantly via loss of wagering capital.
- Tree-Based models performed admirably well, as did Multi-Layered Perceptron and LogisticRegression classifiers.  A realistic expectation for Precision on the Validation subset ranges between 80% and 88%.  Multiple Logistic Regression models also provided coefficient information for determining feature importance.
- This implies that not only is it possible to achieve significant "predictive capability", but we may also retain a realistic level of "interpretability" or explainability with our results if we were to use Tree-Based classifiers.
- Along with DecisionTree and Random Forest models, LogRegression pointed towards the features "pts" and "opp_pts" as the Top 2 features.  While this should seem intuitive and obvious, the interesting aspect of feature importance is that all three models pointed to differing features in order to round out their respective Top 5 lists.
- In the future, it is recommended to explore applications of ML clustering on this dataset to support an increase in the predictive power of classification models.

#### Applications:

- For the purposes of placing "Futures" wagers for NBA Champ, Conf Champ, etc...the first step is to ensure that Team X will qualify for the playoffs.  If a wager is placed, and Team X fails to reach the playoffs, then the ticket is now an expensive piece of kindling.  This project deals ONLY WITH THIS ASPECT of the process: will Team X qualify for the playoffs?
- Further evaluation is necessary to compare the probability that Team X wins the NBA Championship to the payoff odds offered by the sportsbook.  Much information is publicly available to make this evaluation such that a wager can be made with a Positive Expected Value.  It is outside the scope of this project to elaborate further on the topic of sports wagering theory and practice.  As a starting point, please visit the "Wizard of Odds", Michael Shackleford's website at https://wizardofodds.com/games/sports-betting.


#### Instructions for those who wish to reproduce this work or simply follow along:
You Will Need (ALL files must be placed in THE SAME FOLDER!):
- 1. final_nba_project.ipynb file from this git repo
- 2. wranglerer.py file from this git repo
- 3. modeling.py file from this git repo
- 4. nba.csv from this git repo

Ensure:
- CATboost library required in the working environment, however, the code in the Final_Notebook can be removed or commented out in order to run the notebook.
- All files are in the SAME FOLDER
- wranglerer.py and modeling.py each have the .py extension in the file name

Any further assistance required, please email me at myemailaddress@somecompany.com.
