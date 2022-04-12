def coefficients(self, labels, intercept=True):
    # returns dictionary with coefficients as values and labels as keys
    coefs = dict(zip(labels, self.coef_))
    if intercept:
        coefs['intercept'] = self.intercept_
    return coefs
