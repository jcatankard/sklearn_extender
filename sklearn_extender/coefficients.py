def coefficients(self, labels: list, intercept: bool = True) -> dict:
    # returns dictionary with coefficients as values and labels as keys
    coefs = dict(zip(labels, self.coef_))
    if intercept:
        coefs['intercept'] = self.intercept_
    return coefs
