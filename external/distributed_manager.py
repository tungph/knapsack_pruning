import logging

from torch.distributed import get_rank, get_world_size, new_group, barrier

logger = logging.getLogger(__name__)


class DistributedManager():
    distributed = False
    grp = None
    local_rank = 0

    @staticmethod
    def set_args(args):
        DistributedManager.distributed = args.distributed
        if args.distributed:
            logger.info(
                f"[{get_rank()}] >> Setting barrier for group: {list(range(0, get_world_size()))} "
            )
            DistributedManager.grp = new_group(list(range(0, get_world_size())))
            DistributedManager.local_rank = args.local_rank

    @staticmethod
    def is_master():
        return (not DistributedManager.distributed) or DistributedManager.local_rank == 0

    @staticmethod
    def get_rank_():
        return 0 if not DistributedManager.distributed else get_rank()

    @staticmethod
    def is_first():
        return DistributedManager.get_rank_() == 0

    @staticmethod
    def set_barrier():
        if DistributedManager.distributed:
            logger.info(f"[{get_rank()}] >> Barrier waiting ")
            barrier(group=DistributedManager.grp)
            logger.info(f"[{get_rank()}] >> Barrier passed ")
