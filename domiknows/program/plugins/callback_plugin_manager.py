"""
Callback Plugin Manager

Central manager for coordinating callback plugins in training experiments.
"""


class CallbackPluginManager:
    """Manager for registering and coordinating callback plugins."""
    
    def __init__(self):
        self.plugins = []
        self.plugin_names = []
    
    def register(self, plugin, name=None):
        """
        Register a callback plugin.
        
        Args:
            plugin: Plugin instance (must have configure method)
            name: Optional name for the plugin (defaults to class name)
        """
        if name is None:
            name = plugin.__class__.__name__
        
        self.plugins.append(plugin)
        self.plugin_names.append(name)
        
        return self
    
    def add_arguments_to_parser(self, parser):
        """
        Add arguments from all registered plugins to argument parser.
        
        Args:
            parser: argparse.ArgumentParser instance
        """
        for plugin, name in zip(self.plugins, self.plugin_names):
            if hasattr(plugin, 'add_arguments'):
                plugin.add_arguments(parser)
        
        return self
    
    def configure_all(self, **kwargs):
        """
        Configure all registered plugins.
        
        Args:
            **kwargs: Arguments to pass to each plugin's configure method
                     Common args: program, models, args, dataset, optimizer_factory
        """
        for plugin, name in zip(self.plugins, self.plugin_names):
            if hasattr(plugin, 'configure'):
                try:
                    # Different plugins need different arguments
                    # Try to call configure with available kwargs
                    import inspect
                    sig = inspect.signature(plugin.configure)
                    plugin_kwargs = {}
                    
                    for param_name in sig.parameters:
                        if param_name in kwargs:
                            plugin_kwargs[param_name] = kwargs[param_name]
                    
                    plugin.configure(**plugin_kwargs)
                except Exception as e:
                    print(f"[Plugin Manager] Warning: Failed to configure {name}: {e}")
        
        return self
    
    def log_all_configs(self, args):
        """
        Log configuration for all registered plugins.
        
        Args:
            args: Parsed arguments
        """
        print("\n[Active Callbacks]")
        
        for plugin, name in zip(self.plugins, self.plugin_names):
            if hasattr(plugin, 'log_config'):
                try:
                    plugin.log_config(args)
                except Exception as e:
                    print(f"  {name}: Error logging config: {e}")
        
        return self
    
    def final_display_all(self, **kwargs):
        """
        Display final summaries for all registered plugins.
        
        Args:
            **kwargs: Optional arguments to pass to plugins (e.g., final_eval)
        """
        print("\n" + "=" * 60)
        print("CALLBACK PLUGIN SUMMARIES")
        print("=" * 60)
        
        for plugin, name in zip(self.plugins, self.plugin_names):
            if hasattr(plugin, 'final_display'):
                try:
                    import inspect
                    sig = inspect.signature(plugin.final_display)
                    
                    if len(sig.parameters) > 0:
                        # Plugin expects arguments
                        plugin_kwargs = {}
                        for param_name in sig.parameters:
                            if param_name in kwargs:
                                plugin_kwargs[param_name] = kwargs[param_name]
                        plugin.final_display(**plugin_kwargs)
                    else:
                        # Plugin expects no arguments
                        plugin.final_display()
                        
                except Exception as e:
                    print(f"\n[{name}] Error in final display: {e}")
        
        print("\n" + "=" * 60)
        
        return self
    
    def get_plugin(self, name):
        """
        Get a specific plugin by name.
        
        Args:
            name: Plugin name (class name or custom name)
            
        Returns:
            Plugin instance or None if not found
        """
        for plugin, plugin_name in zip(self.plugins, self.plugin_names):
            if plugin_name == name:
                return plugin
        return None
    
    def __len__(self):
        """Return number of registered plugins."""
        return len(self.plugins)
    
    def __repr__(self):
        """String representation of the manager."""
        return f"CallbackPluginManager(plugins={self.plugin_names})"


def create_standard_plugin_manager():
    """
    Create a plugin manager with all standard plugins registered.
    
    Returns:
        CallbackPluginManager with standard plugins
    """
    from .epoch_logging_plugin import EpochLoggingPlugin
    from .adaptive_tnorm_plugin import AdaptiveTNormPlugin
    from .gradient_flow_plugin import GradientFlowPlugin
    from .counting_schedule_plugin import CountingSchedulePlugin
    from .gumbel_monitoring_plugin import GumbelMonitoringPlugin
    from .bert_unfreezing_plugin import BERTUnfreezingPlugin
    
    manager = CallbackPluginManager()
    
    # Register all standard plugins
    manager.register(EpochLoggingPlugin(), 'EpochLogging')
    manager.register(AdaptiveTNormPlugin(), 'AdaptiveTNorm')
    manager.register(GradientFlowPlugin(), 'GradientFlow')
    #anager.register(CountingSchedulePlugin(), 'CountingSchedule')
    manager.register(GumbelMonitoringPlugin(), 'GumbelMonitoring')
    manager.register(BERTUnfreezingPlugin(), 'BERTUnfreezing')
    
    return manager