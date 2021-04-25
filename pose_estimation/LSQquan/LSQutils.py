import logging

from .LSQfunc import *
from .quantizer import *


def quantizer(default_cfg, this_cfg=None):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v

    if target_cfg['bit'] is None:
        q = IdentityQuan
    elif target_cfg['mode'] == 'lsq':
        q = LsqQuan
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['mode'])

    target_cfg.pop('mode')
    return q(**target_cfg)


# def find_modules_to_quantize(model, quan_scheduler):
#     replaced_modules = dict()
#     for name, module in model.named_modules():
#         if type(module) in QuanModuleMapping.keys():
#             if name in quan_scheduler['excepts'].keys():
#                 replaced_modules[name] = QuanModuleMapping[type(module)](
#                     module,
#                     quan_w_fn=quantizer(quan_scheduler['weight'],
#                                         quan_scheduler['excepts'][name]['weight']),
#                     quan_a_fn=quantizer(quan_scheduler['act'],
#                                         quan_scheduler['excepts'][name]['weight'])
#                 )
#             else:
#                 replaced_modules[name] = QuanModuleMapping[type(module)](
#                     module,
#                     quan_w_fn=quantizer(quan_scheduler['weight']),
#                     quan_a_fn=quantizer(quan_scheduler['act'])
#                 )
#         elif name in quan_scheduler['excepts'].keys():
#             logging.warning('Cannot find module %s in the model, skip it' % name)

#     return replaced_modules

def find_modules_to_quantize(model, quan_schedulerA,quan_schedulerW,quan_schedulerE):
    replaced_modules = dict()
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            if( (name in quan_schedulerE) or name == 'features.0.0'):
                tmpA={'bit':None,'all_positive': False}
                tmpW={'bit':None}
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_schedulerW,
                                        tmpW),
                    quan_a_fn=quantizer(quan_schedulerA,
                                        tmpA)
                )
                # replaced_modules[name] = QuanModuleMapping[type(module)](
                #     module,
                #     quan_w_fn=quantizer(quan_schedulerW,
                #                         quan_schedulerE[name].weight),
                #     quan_a_fn=quantizer(quan_schedulerA,
                #                         quan_schedulerE[name].act)
                # )
            else:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_schedulerW),
                    quan_a_fn=quantizer(quan_schedulerA)
                )
        elif name in quan_schedulerE:
            logging.warning('Cannot find module %s in the model, skip it' % name)

    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model
