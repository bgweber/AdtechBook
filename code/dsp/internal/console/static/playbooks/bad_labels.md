# Bad labels: the install label rate check failed

**Owner:** ML team. **Severity:** ticket; it stops the pipeline.

## What it means

A dataset build found a day whose installs per impression are more than twice the median of the other days. Real performance doesn't jump like that in a day; labels do, when an MMP setting changes, postbacks are duplicated, or an experiment's labels leak in. A model trained on them predicts too many installs and overbids.

## Confirm it

Look at the install postbacks received around that day, by the fields the MMP sends, on the Controls tab under **Quarantine**, or:

```bash
docker compose exec trainer python -m adtrain quarantine inspect --ledger /out/ledger --day 41
docker compose exec trainer python -m adtrain quarantine inspect --ledger /out/ledger --day 41 --field touch_type
```

A batch of bad labels usually has a mark: a new value of a field, such as `match_type=probabilistic`, in a block of hours.

## Safe first actions: quarantine and retrain

1. **Quarantine** the bad postbacks, by source, time range, and the field that marks them, on the Controls tab or:

   ```bash
   docker compose exec trainer python -m adtrain quarantine add --datasets /out/datasets \
     --day 41 --where match_type=probabilistic --reason "probabilistic matching turned on by mistake"
   ```

2. **Rebuild** the datasets: every build reads the quarantine list and skips what it marks. Run the pipeline from the Conversions tab, or build a dataset and check that the install label rate check passes.
3. **Retrain from a known-good point**, and compare the new model with the champion trained before the bad day.
4. **Rebuild what carries it forward.** The bidder's device profiles counted the false installs, so the `device_prior_installs` feature is wrong for those devices until it's recounted:

   ```bash
   curl -s -X POST localhost:8081/api/profiles/recount
   ```

   Until then, the bidder prices those devices with a count that includes the false installs, and logs it on every bid, while the rebuilt datasets don't: training and serving disagree. The datasets' "backfill matches logged" check leaves out rows after a device's quarantined postbacks, so it can't tell you whether you recounted; the recount's reply says how many devices changed.

## Escalate

To the MMP integration owner, with the field, the hours, and the count, so the setting is fixed at the source.

## Verify recovery

The rebuilt dataset passes every check, the pipeline's challenger passes, and new days' install rates are back with the median.
