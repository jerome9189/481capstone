class FNCRelativeScore(Metric):
    def __init__(self, name='fnc_relative_score', **kwargs):
        super(FNCRelativeScore, self).__init__(name=name, **kwargs)
        self.max_score = 0.0
        self.cur_score = 0.0

    def update_state(self, y_true, y_pred, sample_weight=1.0):
        if y_true == 'unrelated':
            self.max_score += 0.25 * sample_weight
            if y_pred == 'unrelated':
                self.cur_score += 0.25 * sample_weight
        else:
            self.max_score += 1.0 * sample_weight
            if y_pred == 'agree' or y_pred == 'disagree' or y_pred == 'discuss':
                self.cur_score += 0.25 * sample_weight
            if y_pred == y_true:
                self.cur_score += 0.75 * sample_weight

    def result(self):
        return (self.cur_score/self.max_score)