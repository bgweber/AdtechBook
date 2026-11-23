"""Training for the reference DSP's click model (Chapters 9 and 11).

The pipeline reads the bidder's ledger, builds click examples, trains a
logistic regression with FTRL on hashed tokens, validates it out of time,
and publishes a model artifact the Go bidder can load:

    ledger -> examples -> tokens -> FTRL -> metrics + checks -> artifact

Run it from the command line (`python -m adtrain train`), or as a small
service the bidder's console calls (`python -m adtrain serve`).
"""

__version__ = "0.8.0"
