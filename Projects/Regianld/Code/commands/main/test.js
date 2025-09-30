const {
    SlashCommandBuilder,
    EmbedBuilder,
    MessageFlags
} = require('discord.js');

module.exports = {
    cooldown: 5,
    data: new SlashCommandBuilder()
        .setName('test')
        .setDescription('Hello World!'),

    async execute(interaction) {
        await interaction.reply(`Oi!`)
    }
};