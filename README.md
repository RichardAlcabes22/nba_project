# nba_classification_project
#### Welcome to this initial exploration of NBA Team Data from the 1979-80 season until the current 2022-23 season!  The primary purpose of this project is to make accurate mid-season predictions about which teams will make the playoffs at the end of the season. The project data-set was personally curated coutesy of https://www.basketball-reference.com/.  Season-wide Team data for each NBA Team from 1979 until present was aggregated and each row was labeled as a "Playoff" team or a "Non-Playoff" team.
#### The goals of this initial exploration are as follows:
- PRIMARY: Generate actionable predictions regarding whether or not a team will qualify for the Playoffs at season's-end.
- SECONDARY: Draw reasonable inferences regarding the characteristics of those teams which qualify for the playoffs, and those that do not.
- SECONDARY: Identify those features which retain the greatest importance for determining the potential playoff status of a particular team.

#### PROJECT DESCRIPTION and ASSUMPTIONS:
- Basketball has changed significantly over the past several generations, with respect to tactics, strategy, and tempo.  Thus, the team statistics generated per game have changed over time to reflect this assertion. 

- Consequently, this dataset does not display team stats on a "per game" basis.  Instead, team stats are displayed "per 100 possessions".  This allows for a somewhat-more-valid comparison to be made between teams of differing generations.  This also partially controls for stats being functionally-dependent upon common in-game conditions and provides a window into a more distilled view of a team's capabilities.

- The 1979-80 NBA Season was chosen as the starting point for the dataset because it represesnts the most recent, fundamental change to the game: the 3-pt Shot.  While, it took over 20 years for the 3-pt shot to actually change the game, it is important to note that the efficacy of the model is assumed to be correlated with the number of datapoints (approx. 1200), and thus, 1979 was chosen over 2003, as this would have only yielded about 600 datapoints.

#### Project Planning:
- Plan: Questions and Hypotheses
- Acquire: Curated dataset from from https://www.basketball-reference.com/ by subsetting team-level statistics which include stats reflecting "skill, talent, and hustle".  For example, team Field Goal % is a function of skill level(among other contributors), while Steals, Rebounds, or Fouls Drawn are a function of "talent and hustle" (among others).
- Prepare: Kept outliers after investigating their nature, missingness was a non-issue, as there were ZERO entries containing NULL values for predictors.  StandardScaler used for scaling purposes.  Split into ML subsets (Train/Validate/Test).
- Explore: Univariate and multi-variate analysis, correlation matrix, 2D visualization, correlation significance testing, 2-sample T-testing for significant differences in means.
- Model: Established a baseline "Precision" for Positive class of 56.5% using the most frequent target occurance of "yes: playoffs".  Then with a Vanilla, Obviously Overfit, DecisionTreeClassifier with default hyperparameters, established a new Precision floor of 82.0%. After trying different tree-based and non-tree-based algorithms with differing tunings, findings indicated a DecisionTree Classifier with a max_depth setting of 4 yielded best results (88.0% Precision on Test).
- Deliver: Please refer to this doc as well as the Final_NBA.ipynb file for the finished version of the presentation, in addition to each of the underlying exploratory notebooks.

#### Initial hypotheses and questions:
* What meaningful features can be leveraged to create a model that predicts whether or not an NBA team will qualify for the playoffs?  
* Can statistics which reflect a team's level of skill, preparation and effort (hustle) yield similar or even better results than the traditional statistics upon which most players/teams are evaluated?  
* Can these chosen features transcend the changing nature of the game in order to allow for an "apples-to-apples" categorization of teams from multiple eras of NBA play? 
* If all of the above are realistic and possible, can an informed sports-wager be placed utilizing the outputs of said modeling?
* If wagers are at-risk, which is more benign, a false positive: a team predicted to make the playoffs but fails to do so, or a false-negative: a team that is predicted to NOT make the playoffs but ulyimately does?
* Can the sports-wagering public make use of the model output when placing "futures" wagers on teams that may or may not win NBA Championship, Conf Championship, or Div Championship?

#### Data Dictionary: 

|Feature |  Data type | Definition |
|---|---|---|
| fg_pct: | float | percentage of shots taken which scored | skill |
| opp_fg_pct: | float | same as above but for the *opponent* | hustle |
| three_pt_pct: | float | percentage of 3-point shots taken which scored | skill |
| opp_three_pt_pct: | float | same as above but for the *opponent* | preparation |
| ft_pct: | float | percentage of free throws taken which scored | skill |
| rebounds: | float | number of missed shots recovered from rim/backboard | hustle |
| opp_rebounds: | float | same as above but for the *opponent* | hustle |
| assists: | float | number of passes completed to player in the act of scoring a field goal | skill |
| steals: | float | number of times the ball was successfully stripped from opposing player | hustle |
| opp_steals: | float | same as above but for the *opponent* | skill |
| trnovers_committed: | float | number of times the ball was turned-over to the opponent | skill |
| pts: | float | number of points scored | preparation |
| opp_pts: | float | same as above but for the *opponent* | preparation |
| opp_fouls: | float | number of times the opponent fouled a player | hustle |
| prev-season: | int | 1-made Playoffs in Previous Season / 0-did not make Playoffs in Previous Season | |
| playoffs: | int | TARGET: 1-Playoff Team / 0-Non-Playoff Team | |