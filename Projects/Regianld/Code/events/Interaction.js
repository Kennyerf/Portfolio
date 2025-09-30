const { Events, MessageFlags, Collection } = require('discord.js');

module.exports = {
	name: Events.InteractionCreate,
	async execute(interaction) {
		if (!interaction.isChatInputCommand()) return;

		const command = interaction.client.commands.get(interaction.commandName);
		if (!command) return console.error(`No command matching ${interaction.commandName} was found.`);

		// Ensure cooldown collection exists for the command.
		if (!interaction.client.cooldowns) interaction.client.cooldowns = new Collection();
		if (!interaction.client.cooldowns.has(command.data.name))
			interaction.client.cooldowns.set(command.data.name, new Collection());

		const now = Date.now();
		const timestamps = interaction.client.cooldowns.get(command.data.name);
		const cooldown = (command.cooldown ?? 3) * 1000;

		if (timestamps.has(interaction.user.id)) {
			const expiration = timestamps.get(interaction.user.id) + cooldown;
			if (now < expiration)
				return interaction.reply({
					content: `Please wait before using \`${command.data.name}\` again (<t:${Math.round(expiration / 1000)}:R>).`,
					flags: MessageFlags.Ephemeral,
				});
		}

		timestamps.set(interaction.user.id, now);
		setTimeout(() => timestamps.delete(interaction.user.id), cooldown);

		try {
			await command.execute(interaction);
		} catch (error) {
			console.error(error);
			const errorReply = {
				content: 'There was an error executing this command!',
				flags: MessageFlags.Ephemeral,
			};
			interaction.replied || interaction.deferred
				? await interaction.followUp(errorReply)
				: await interaction.reply(errorReply);
		}
	},
};
