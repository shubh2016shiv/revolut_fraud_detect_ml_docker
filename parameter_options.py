class Options:
    options = {
        "Has Email": ('YES', 'NO'),
        "User Status": ("ACTIVE", "LOCKED"),
        "KYC": ("PASSED", "PENDING", "NONE", "FAILED"),
        "Currency": ("EUR", "GBP", "USD"),
        "Entry Method": ("chip", "cont", "mags", "manu", "mcon", "misc"),
        "Type": ("ATM", "BANK_TRANSFER", "CARD_PAYMENT", "P2P", "TOPUP"),
        "Merchant Country": [line.strip() for line in open("./artifacts/merchant_country", "r").readlines()],
        "Merchant Category": [line.strip() for line in open("./artifacts/merchant_category", "r").readlines()],
        "Transaction Status": ("CANCELLED","COMPLETED","DECLINED","FAILED","PENDING","RECORDED","REVERTED"),
        "Source": ("APOLLO","BRIZO","CRONUS","GAIA","HERA","INTERNAL","LETO","LIMOS","MINOS","NYX","OPHION"),
        "Country": [line.strip() for line in open("./artifacts/country_names", "r").readlines()],
        "Phone Country": [line.strip() for line in open("./artifacts/phone_country", "r").readlines()]
    }

    @classmethod
    def get_options(cls, option):
        if option in cls.options:
            return cls.options[option]
        else:
            return []
