import logging
from typing import List

from clt_observer import ObservedAlignedSeq
from barcode_metadata import BarcodeMetadata

def binarize_observations(bcode_meta: BarcodeMetadata, observations: List[ObservedAlignedSeq]):
    """
    Prepares the observations to be sent to phylip
    Each event is represented by a tuple (start idx, end idx, insertion)

    @return processed_seqs: Dict[str, List[float, List[List[Event]], CellState]]
                this maps the sequence names to event list and abundance
            all_event_dict: List[Dict[event_tuple, event number]]
                maps events to their event number
            event_list: List[event_tuple]
                the reverse of all_event_dict
    """
    # Figure out what events happened
    processed_seqs = {}
    all_events = [set() for _ in range(bcode_meta.num_barcodes)]
    for idx, obs in enumerate(observations):
        evts_list = []
        for bcode_idx, allele_evts in enumerate(obs.allele_events_list):
            evts = allele_evts.events
            evts_bcode = [evt for evt in evts]
            all_events[bcode_idx].update(evts_bcode)
            evts_list.append(evts_bcode)
        processed_seqs["seq{}".format(idx)] = [obs.abundance, evts_list, obs.cell_state]
        logging.info("seq%d %s", idx, str(obs))

    # Assemble events in a dictionary
    event_dicts = []
    event_list = []
    num_evts = 0
    for bcode_idx, bcode_evts in enumerate(all_events):
        bcode_evt_list = list(bcode_evts)
        event_list += [(bcode_idx, evt) for evt in bcode_evt_list]
        event_bcode_dict = {evt: num_evts + i for i, evt in enumerate(bcode_evt_list)}
        num_evts += len(event_bcode_dict)
        event_dicts.append(event_bcode_dict)

    return processed_seqs, event_dicts, event_list
